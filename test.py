import argparse
import importlib.util
import json
import logging
import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader


# Globals populated after loading the training utilities.
AudioDataset = None
DataConfig = None
ModelConfig = None
Wav2Vec2Classifier = None
aggregate_subject_predictions = None
bootstrap_metric_ci = None
collate_fn = None
compute_threshold_metrics = None
evaluate = None
find_best_threshold = None
plot_calibration_reliability = None
plot_confusion_matrices = None
plot_decision_curve = None
plot_pr_curve_with_ci = None
plot_probability_distributions = None
plot_roc_curve_with_ci = None
plot_sensitivity_specificity_bar = None
plot_threshold_diagnostics = None
scan_thresholds = None


def resolve_train_module_path(provided: Optional[str]) -> Path:
    """Resolve the path to train.py, supporting notebook and CLI usage."""

    if provided:
        candidate = Path(provided).expanduser().resolve()
        if candidate.is_dir():
            candidate = candidate / "train.py"
        if not candidate.exists():
            raise FileNotFoundError(f"Specified training module path does not exist: {candidate}")
        return candidate

    env_path = os.environ.get("TRAIN_MODULE_PATH")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.is_dir():
            candidate = candidate / "train.py"
        if candidate.exists():
            return candidate

    candidate_dirs: List[Path] = []
    if "__file__" in globals():
        candidate_dirs.append(Path(__file__).resolve().parent)
    candidate_dirs.append(Path.cwd())
    for entry in sys.path:
        try:
            candidate_dirs.append(Path(entry))
        except Exception:
            continue

    seen: Set[Path] = set()
    for base in candidate_dirs:
        try:
            current = base.resolve()
        except FileNotFoundError:
            continue
        for parent in [current, *current.parents]:
            if parent in seen:
                continue
            seen.add(parent)
            for name in ("train.py", "wavlm_large.ipynb"):
                candidate = parent / name
                if candidate.exists():
                    return candidate.resolve()

    # As a final fallback, inspect immediate child directories for a train.py file
    for base in candidate_dirs:
        try:
            resolved_base = base.resolve()
        except FileNotFoundError:
            continue
        try:
            for child in resolved_base.iterdir():
                if not child.is_dir():
                    continue
                for name in ("train.py", "wavlm_large.ipynb"):
                    candidate = child / name
                    if candidate.exists():
                        return candidate.resolve()
        except (PermissionError, FileNotFoundError):
            continue

    raise FileNotFoundError(
        "Unable to locate training utilities (train.py or wavlm_large.ipynb). "
        "Provide --train-module, set TRAIN_MODULE_PATH, or place test.py near the training notebook."
    )


def _load_notebook_module(notebook_path: Path, module_name: str = "train_module"):
    """Execute the first three code cells of a notebook as a Python module."""

    try:
        import nbformat
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "nbformat is required to load training utilities from a notebook. "
            "Install nbformat or provide a train.py file."
        ) from exc

    nb = nbformat.read(notebook_path, as_version=4)
    module = types.ModuleType(module_name)
    module.__file__ = str(notebook_path)
    sys.modules[module_name] = module

    code_cells = [cell for cell in nb.cells if cell.get("cell_type") == "code"]
    if not code_cells:
        raise RuntimeError(f"Notebook {notebook_path} contains no code cells to execute.")

    # Execute the first three code cells (indices 0-2) to mirror the training setup.
    for idx, cell in enumerate(code_cells[:3]):
        source = cell.get("source", "")
        if not source.strip():
            continue
        exec(compile(source, f"{notebook_path}#cell{idx}", "exec"), module.__dict__)

    return module


def load_training_utilities(train_path: Path) -> None:
    """Dynamically import the training helpers from the resolved training source."""

    global AudioDataset
    global DataConfig
    global ModelConfig
    global Wav2Vec2Classifier
    global aggregate_subject_predictions
    global bootstrap_metric_ci
    global collate_fn
    global compute_threshold_metrics
    global evaluate
    global find_best_threshold
    global plot_calibration_reliability
    global plot_confusion_matrices
    global plot_decision_curve
    global plot_pr_curve_with_ci
    global plot_probability_distributions
    global plot_roc_curve_with_ci
    global plot_sensitivity_specificity_bar
    global plot_threshold_diagnostics
    global scan_thresholds

    if train_path.suffix == ".ipynb":
        module = _load_notebook_module(train_path)
    else:
        spec = importlib.util.spec_from_file_location("train_module", train_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load training utilities from {train_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["train_module"] = module
        spec.loader.exec_module(module)

    AudioDataset = module.AudioDataset
    DataConfig = module.DataConfig
    ModelConfig = module.ModelConfig
    Wav2Vec2Classifier = module.Wav2Vec2Classifier
    aggregate_subject_predictions = module.aggregate_subject_predictions
    bootstrap_metric_ci = module.bootstrap_metric_ci
    collate_fn = module.collate_fn
    compute_threshold_metrics = module.compute_threshold_metrics
    evaluate = module.evaluate
    find_best_threshold = module.find_best_threshold
    plot_calibration_reliability = module.plot_calibration_reliability
    plot_confusion_matrices = module.plot_confusion_matrices
    plot_decision_curve = module.plot_decision_curve
    plot_pr_curve_with_ci = module.plot_pr_curve_with_ci
    plot_probability_distributions = module.plot_probability_distributions
    plot_roc_curve_with_ci = module.plot_roc_curve_with_ci
    plot_sensitivity_specificity_bar = module.plot_sensitivity_specificity_bar
    plot_threshold_diagnostics = module.plot_threshold_diagnostics
    scan_thresholds = module.scan_thresholds


@dataclass
class CMDCConfig:
    data_dir: str = "./CMDC"
    overlap_ratio: float = 0.2
    segment_duration: int = 7
    sample_rate: int = 16000


@dataclass
class EvalConfig:
    model_path: str = "best_model_wavlm_lagre.pt"
    output_dir: str = "logs_wavlm_large/cmdc_external"
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 24
    threshold: Optional[float] = None
    beta: float = 2.0
    device: Optional[str] = None
    local_files_only: bool = False


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "cmdc_eval.log")
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=[logging.StreamHandler()],
        )
    file_handler_present = any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == os.path.abspath(log_path)
        for handler in root_logger.handlers
    )
    if not file_handler_present:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(log_format, date_format))
        root_logger.addHandler(fh)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(path: str) -> Tuple[int, str]:
    base = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(ch for ch in base if ch.isdigit())
    return (int(digits) if digits else float("inf"), base.lower())


def load_cmdc_metadata(cfg: CMDCConfig) -> Tuple[List[str], List[int], List[str]]:
    data_dir = cfg.data_dir
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"CMDC directory not found: {data_dir}")

    label_map = {"HC": 0, "MDD": 1}
    file_paths: List[str] = []
    labels: List[int] = []
    subjects: List[str] = []

    for subject in sorted(os.listdir(data_dir)):
        subject_path = os.path.join(data_dir, subject)
        if not os.path.isdir(subject_path):
            continue
        subject_upper = subject.upper()
        label: Optional[int]
        if subject_upper.startswith("HC"):
            label = label_map["HC"]
        elif subject_upper.startswith("MDD"):
            label = label_map["MDD"]
        else:
            logging.warning("Skipping subject %s (label not inferred)", subject)
            continue
        wavs: List[str] = []
        for root, _, files in os.walk(subject_path):
            for filename in files:
                if filename.lower().endswith(".wav"):
                    wavs.append(os.path.join(root, filename))
        if not wavs:
            logging.warning("No .wav files found for subject %s", subject)
            continue
        for wav_path in sorted(wavs, key=natural_key):
            file_paths.append(wav_path)
            labels.append(label)
            subjects.append(subject)

    if not file_paths:
        raise RuntimeError(f"No wav files discovered under {data_dir}")

    label_counts = {"HC": labels.count(0), "MDD": labels.count(1)}
    logging.info(
        "CMDC metadata | subjects=%d | files=%d | HC_files=%d | MDD_files=%d",
        len(set(subjects)),
        len(file_paths),
        label_counts["HC"],
        label_counts["MDD"],
    )
    return file_paths, labels, subjects


def build_cmdc_dataset(
    file_paths: List[str],
    labels: List[int],
    subjects: List[str],
    data_cfg: DataConfig,
) -> AudioDataset:
    data = {"paths": file_paths, "labels": labels, "subjects": subjects}
    samples = AudioDataset.build_samples(data, data_cfg)
    logging.info(
        "Segment statistics | total=%d | HC=%d | MDD=%d",
        len(samples),
        sum(sample["label"] == 0 for sample in samples),
        sum(sample["label"] == 1 for sample in samples),
    )
    return AudioDataset(samples, data_cfg, split="external")


def create_dataloader(dataset: AudioDataset, data_cfg: DataConfig, model_cfg: ModelConfig, eval_cfg: EvalConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=eval_cfg.batch_size,
        shuffle=False,
        num_workers=eval_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: collate_fn(batch, data_cfg, model_cfg),
    )


def prepare_model(model_cfg: ModelConfig, eval_cfg: EvalConfig) -> Wav2Vec2Classifier:
    device = torch.device(eval_cfg.device) if eval_cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2Classifier(model_cfg, num_classes=2).to(device)
    if not os.path.exists(eval_cfg.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {eval_cfg.model_path}")
    state = torch.load(eval_cfg.model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.warning("Missing keys when loading state_dict: %s", missing)
    if unexpected:
        logging.warning("Unexpected keys when loading state_dict: %s", unexpected)
    model.eval()
    return model


def evaluate_segments(
    model: Wav2Vec2Classifier,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    with torch.no_grad():
        segment_metrics, raw_outputs = evaluate(
            model,
            dataloader,
            device,
            use_amp=False,
            split_name="CMDC-Segments",
        )
    return segment_metrics, raw_outputs


def subject_level_analysis(
    subject_labels: np.ndarray,
    subject_probabilities: np.ndarray,
    output_dir: str,
    beta: float,
    provided_threshold: Optional[float] = None,
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    if subject_labels.size == 0:
        logging.warning("No subject-level predictions available; skipping detailed analysis.")
        return results

    if len(np.unique(subject_labels)) > 1:
        auc_value = roc_auc_score(subject_labels, subject_probabilities)
        pr_auc_value = average_precision_score(subject_labels, subject_probabilities)
        auc_ci = bootstrap_metric_ci(subject_labels, subject_probabilities, roc_auc_score)
        pr_auc_ci = bootstrap_metric_ci(subject_labels, subject_probabilities, average_precision_score)
    else:
        auc_value = float("nan")
        pr_auc_value = float("nan")
        auc_ci = (float("nan"), float("nan"))
        pr_auc_ci = (float("nan"), float("nan"))

    if provided_threshold is None:
        best_threshold, threshold_scan = find_best_threshold(subject_labels, subject_probabilities, beta=beta)
        logging.info("Best threshold determined on CMDC via F%.1f maximisation: %.3f", beta, best_threshold)
    else:
        best_threshold = provided_threshold
        threshold_scan = scan_thresholds(subject_labels, subject_probabilities, np.linspace(0.01, 0.99, 99), beta=beta)
        logging.info("Using provided threshold: %.3f", best_threshold)

    threshold_metrics = compute_threshold_metrics(subject_labels, subject_probabilities, best_threshold)

    # Bootstrap confidence intervals for threshold-dependent metrics
    def make_metric_fn(metric_name: str):
        def _metric(labels: np.ndarray, probs: np.ndarray) -> float:
            preds = (probs >= best_threshold).astype(int)
            if metric_name == "f1":
                return f1_score(labels, preds, zero_division=0)
            if metric_name == "mcc":
                return matthews_corrcoef(labels, preds) if len(np.unique(labels)) > 1 else 0.0
            if metric_name == "balanced_accuracy":
                return balanced_accuracy_score(labels, preds)
            if metric_name == "precision":
                return (preds[labels == 1].sum() / preds.sum()) if preds.sum() else 0.0
            if metric_name == "recall":
                positives = (labels == 1)
                return preds[positives].sum() / positives.sum() if positives.any() else 0.0
            return float("nan")

        return _metric

    f1_ci = bootstrap_metric_ci(subject_labels, subject_probabilities, make_metric_fn("f1"))
    mcc_ci = bootstrap_metric_ci(subject_labels, subject_probabilities, make_metric_fn("mcc"))
    bal_acc_ci = bootstrap_metric_ci(subject_labels, subject_probabilities, make_metric_fn("balanced_accuracy"))

    results.update(
        {
            "auc": float(auc_value),
            "auc_ci_low": float(auc_ci[0]),
            "auc_ci_high": float(auc_ci[1]),
            "pr_auc": float(pr_auc_value),
            "pr_auc_ci_low": float(pr_auc_ci[0]),
            "pr_auc_ci_high": float(pr_auc_ci[1]),
            "threshold": float(best_threshold),
            "accuracy": float(threshold_metrics.get("accuracy", float("nan"))),
            "precision": float(threshold_metrics.get("precision", float("nan"))),
            "recall": float(threshold_metrics.get("recall", float("nan"))),
            "specificity": float(threshold_metrics.get("specificity", float("nan"))),
            "npv": float(threshold_metrics.get("npv", float("nan"))),
            "balanced_accuracy": float(threshold_metrics.get("balanced_accuracy", float("nan"))),
            "balanced_accuracy_ci_low": float(bal_acc_ci[0]),
            "balanced_accuracy_ci_high": float(bal_acc_ci[1]),
            "f1": float(threshold_metrics.get("f1", float("nan"))),
            "f1_ci_low": float(f1_ci[0]),
            "f1_ci_high": float(f1_ci[1]),
            "mcc": float(threshold_metrics.get("mcc", float("nan"))),
            "mcc_ci_low": float(mcc_ci[0]),
            "mcc_ci_high": float(mcc_ci[1]),
            "youden_j": float(threshold_metrics.get("youden_j", float("nan"))),
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    roc_path = os.path.join(output_dir, "cmdc_subject_roc.png")
    pr_path = os.path.join(output_dir, "cmdc_subject_pr.png")
    threshold_path = os.path.join(output_dir, "cmdc_subject_threshold.png")
    bar_path = os.path.join(output_dir, "cmdc_subject_sens_spec.png")
    cm_path = os.path.join(output_dir, "cmdc_subject_confusion.png")
    calib_path = os.path.join(output_dir, "cmdc_subject_calibration.png")
    dca_path = os.path.join(output_dir, "cmdc_subject_dca.png")
    dist_path = os.path.join(output_dir, "cmdc_subject_probability.png")

    plot_roc_curve_with_ci(subject_labels, subject_probabilities, roc_path, "MDD")
    plot_pr_curve_with_ci(subject_labels, subject_probabilities, pr_path, "MDD")
    plot_threshold_diagnostics(threshold_scan, best_threshold, threshold_path, beta=beta)
    plot_sensitivity_specificity_bar(threshold_metrics, bar_path)
    plot_confusion_matrices(threshold_metrics["confusion_matrix"], cm_path)
    brier, ece = plot_calibration_reliability(subject_labels, subject_probabilities, calib_path)
    plot_decision_curve(subject_labels, subject_probabilities, dca_path)
    plot_probability_distributions(subject_labels, subject_probabilities, dist_path)

    results["brier"] = float(brier)
    results["ece"] = float(ece)
    results["roc_curve_path"] = roc_path
    results["pr_curve_path"] = pr_path
    results["threshold_curve_path"] = threshold_path
    results["sens_spec_path"] = bar_path
    results["confusion_matrix_path"] = cm_path
    results["calibration_path"] = calib_path
    results["decision_curve_path"] = dca_path
    results["probability_path"] = dist_path

    return results


def _to_serializable(value):
    """Recursively convert numpy objects to JSON-serialisable types."""

    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_metrics(output_dir: str, metrics: Dict[str, Dict[str, float]]) -> None:
    path = os.path.join(output_dir, "cmdc_metrics.json")
    serialisable = _to_serializable(metrics)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, ensure_ascii=False)
    logging.info("Saved metrics to %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External evaluation on CMDC using wavlm-large best model")
    parser.add_argument("--data-dir", default="./CMDC", help="Path to CMDC dataset root")
    parser.add_argument("--model-path", default="best_model_wavlm_lagre.pt", help="Path to the best model checkpoint")
    parser.add_argument("--output-dir", default="logs_wavlm_large/cmdc_external", help="Directory to store evaluation outputs")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--seed", type=int, default=24, help="Random seed")
    parser.add_argument("--threshold", type=float, default=None, help="Optional fixed decision threshold")
    parser.add_argument("--beta", type=float, default=2.0, help="Beta value for F-beta threshold selection")
    parser.add_argument("--device", default=None, help="Computation device (e.g., cuda:0)")
    parser.add_argument("--local-files-only", action="store_true", help="Force transformers to use local files only")
    parser.add_argument("--train-module", default=None, help="Path to train.py (defaults to discovering alongside test.py)")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    eval_cfg = EvalConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        threshold=args.threshold,
        beta=args.beta,
        device=args.device,
        local_files_only=args.local_files_only,
    )

    setup_logging(eval_cfg.output_dir)
    seed_everything(eval_cfg.seed)

    train_module_path = resolve_train_module_path(args.train_module)
    load_training_utilities(train_module_path)

    cmdc_cfg = CMDCConfig(data_dir=args.data_dir)
    file_paths, labels, subjects = load_cmdc_metadata(cmdc_cfg)

    data_cfg = DataConfig(
        data_dir=cmdc_cfg.data_dir,
        sample_rate=cmdc_cfg.sample_rate,
        segment_duration=cmdc_cfg.segment_duration,
        overlap_ratio=cmdc_cfg.overlap_ratio,
        apply_augmentation=False,
    )

    model_cfg = ModelConfig(
        model_name="microsoft/wavlm-large",
        hf_cache_dir="./hf_cache",
        local_files_only=eval_cfg.local_files_only,
        force_offline=False,
    )

    os.makedirs(model_cfg.hf_cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", model_cfg.hf_cache_dir)
    if eval_cfg.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    dataset = build_cmdc_dataset(file_paths, labels, subjects, data_cfg)
    dataloader = create_dataloader(dataset, data_cfg, model_cfg, eval_cfg)

    device = torch.device(eval_cfg.device) if eval_cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = prepare_model(model_cfg, eval_cfg)

    logging.info("Running segment-level inference on CMDC (%d batches)...", len(dataloader))
    segment_metrics, raw_outputs = evaluate_segments(model, dataloader, device)

    logging.info("Segment-level metrics: %s", {k: float(v) if isinstance(v, (int, float)) else v for k, v in segment_metrics.items() if k != "confusion_matrix"})
    if segment_metrics.get("confusion_matrix") is not None:
        logging.info("Segment-level confusion matrix:\n%s", segment_metrics["confusion_matrix"])

    subject_data = aggregate_subject_predictions(
        raw_outputs["labels"].tolist(),
        raw_outputs["probabilities"].tolist(),
        raw_outputs["subjects"].tolist(),
    )

    subject_results = subject_level_analysis(
        subject_data.labels,
        subject_data.probabilities,
        eval_cfg.output_dir,
        beta=eval_cfg.beta,
        provided_threshold=eval_cfg.threshold,
    )
    if subject_results:
        logged_subject = {k: float(v) if isinstance(v, (int, float)) else v for k, v in subject_results.items() if not k.endswith("_path")}
        logging.info("Subject-level metrics: %s", logged_subject)
    else:
        logging.warning("Subject-level metrics unavailable.")

    metrics = {
        "segment": {k: float(v) if isinstance(v, (int, float)) else v for k, v in segment_metrics.items() if k != "confusion_matrix"},
        "segment_confusion_matrix": segment_metrics.get("confusion_matrix"),
        "subject": subject_results,
    }
    save_metrics(eval_cfg.output_dir, metrics)


if __name__ == "__main__":
    main()
