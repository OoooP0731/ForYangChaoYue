import logging
import os
import random
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

try:
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import ReadTimeout as RequestsReadTimeout
except Exception:  # pragma: no cover - requests may be unavailable at runtime
    RequestsConnectionError = tuple()  # type: ignore[assignment]
    RequestsReadTimeout = tuple()  # type: ignore[assignment]


def _is_hf_timeout_error(error: Exception) -> bool:
    timeout_types = (RequestsReadTimeout, RequestsConnectionError)
    if timeout_types != tuple() and isinstance(error, timeout_types):
        return True
    message = str(error).lower()
    return "timed out" in message or "time out" in message


def _enable_hf_offline_mode(reason: str) -> None:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return
    logger.warning("%s Falling back to Hugging Face offline cache.", reason)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

def _safe_model_tag(model_name: str) -> str:
    """Create a filesystem-friendly tag from a Hugging Face model id."""

    return model_name.replace("/", "_").replace(":", "_")


@dataclass
class DataConfig:
    data_dir: str = "./audio_lanzhou_2015"
    sample_rate: int = 16000
    segment_duration: int = 7
    overlap_ratio: float = 0.2
    max_segments_per_subject: Optional[int] = None
    normalize_amplitude: bool = True
    apply_silence_trim: bool = True
    silence_frame_ms: int = 25
    silence_hop_ms: int = 10
    silence_energy_threshold: float = 1e-4
    apply_median_filter: bool = True
    median_filter_kernel: int = 5
    apply_augmentation: bool = True
    augmentation_prob: float = 0.6
    noise_std_range: Tuple[float, float] = (0.001, 0.01)
    gain_range: Tuple[float, float] = (0.9, 1.1)
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    time_stretch_prob: float = 0.3
    time_shift_max_ratio: float = 0.2
    time_shift_prob: float = 0.4
    time_dropout_prob: float = 0.4
    time_dropout_max_ratio: float = 0.1


@dataclass
class ModelConfig:
    model_name: str = "microsoft/wavlm-large"
    hf_cache_dir: str = "./hf_cache"
    local_files_only: bool = False
    force_offline: bool = False
    dropout: float = 0.3
    frame_dropout: float = 0.1
    classifier_hidden_dim: int = 256
    unfreeze_last_n_layers: int = 2

    @property
    def tag(self) -> str:
        return _safe_model_tag(self.model_name)


@dataclass
class TrainConfig:
    batch_size: int = 64
    num_workers: int = 4
    persistent_workers: bool = True
    head_learning_rate: float = 6e-3
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    scheduler_eta_min: float = 1e-6
    max_grad_norm: float = 1.0
    use_amp: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 24
    resume_from_best: bool = False
    best_model_path: str = "best_model_wavlm_lagre.pt"
    log_dir: str = "logs_wavlm_large"
    curve_path: str = "training_curves_wavlm_large.png"
    val_selection_metric: str = "segment"
    log_eval_details: bool = True
    plot_training_curves: bool = True
    max_nan_warnings: int = 3
    nan_lr_scale: float = 0.5
    disable_amp_on_nan: bool = True
    pretrained_freeze_epochs: int = 5
    pretrained_finetune_epochs: int = 20
    label_smoothing: float = 0.1


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = ExperimentConfig()
CONFIG.train.best_model_path = "best_model_wavlm_lagre.pt"
CONFIG.train.log_dir = "logs_wavlm_large"
CONFIG.train.curve_path = "training_curves_wavlm_large.png"


# ---------------------------------------------------------------------------
# Logging and reproducibility utilities
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            format=log_format,
            datefmt=date_fmt,
            level=logging.INFO,
            handlers=[logging.StreamHandler()],
        )
    os.makedirs(CONFIG.train.log_dir, exist_ok=True)
    log_path = os.path.join(CONFIG.train.log_dir, "training.log")
    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == os.path.abspath(log_path)
        for h in root_logger.handlers
    ):
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format, date_fmt))
        root_logger.addHandler(file_handler)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metadata loading and dataset preparation
# ---------------------------------------------------------------------------

def natural_key(path: str) -> Tuple[int, str]:
    base = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(filter(str.isdigit, base))
    return (int(digits) if digits else float("inf"), base.lower())


def load_metadata(data_cfg: DataConfig) -> Tuple[List[str], List[int], List[str]]:
    excel_path = os.path.join(data_cfg.data_dir, "subjects_information_audio_lanzhou_2015.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Metadata file not found: {excel_path}")
    df = pd.read_excel(excel_path)
    subj_col = next((c for c in df.columns if "subject" in c.lower() or c.lower() == "id"), None)
    label_col = next((c for c in df.columns if c.lower() in {"label", "type"}), None)
    if subj_col is None or label_col is None:
        raise KeyError(f"Required columns missing. Available columns: {list(df.columns)}")

    label_map = {"HC": 0, "MDD": 1}
    file_paths: List[str] = []
    labels: List[int] = []
    subjects: List[str] = []

    for _, row in df.iterrows():
        raw_id = str(row[subj_col]).strip()
        subject_id = raw_id.zfill(8) if raw_id.isdigit() else raw_id
        label_name = str(row[label_col]).strip()
        label = label_map.get(label_name)
        if label is None:
            continue
        subject_dir = os.path.join(data_cfg.data_dir, subject_id)
        if not os.path.isdir(subject_dir):
            logger.warning("Subject directory missing: %s", subject_dir)
            continue
        wavs: List[str] = []
        for root, _, files in os.walk(subject_dir):
            wavs.extend(os.path.join(root, f) for f in files if f.lower().endswith(".wav"))
        for wav_path in sorted(wavs, key=natural_key):
            file_paths.append(wav_path)
            labels.append(label)
            subjects.append(subject_id)

    logger.info(
        "Loaded %d audio files from %d subjects (HC=%d, MDD=%d)",
        len(file_paths),
        len(set(subjects)),
        labels.count(0),
        labels.count(1),
    )
    return file_paths, labels, subjects


def split_segments(
    samples: List[Dict],
    train_cfg: TrainConfig,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    total = len(samples)
    if total == 0:
        return [], [], []
    rng = random.Random(train_cfg.random_seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    n_train = int(total * train_cfg.train_ratio)
    n_val = int(total * train_cfg.val_ratio)
    n_train = min(max(n_train, 0), total)
    n_val = min(max(n_val, 0), total - n_train)
    n_test = total - n_train - n_val
    if n_test == 0 and total >= 3:
        n_test = 1
        if n_val > 0:
            n_val -= 1
        elif n_train > 0:
            n_train -= 1
    train_samples = shuffled[:n_train]
    val_samples = shuffled[n_train:n_train + n_val]
    test_samples = shuffled[n_train + n_val:]
    return train_samples, val_samples, test_samples


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------

def summarize_split_counts(split_samples: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Return a DataFrame summarising HC/MDD segment counts per split."""

    rows: List[Dict[str, object]] = []
    label_names = {0: "HC", 1: "MDD"}
    for split_name, samples in split_samples.items():
        label_counter = Counter(sample["label"] for sample in samples)
        row = {"split": split_name, "total_segments": len(samples)}
        for label_id, label_name in label_names.items():
            row[label_name] = label_counter.get(label_id, 0)
        rows.append(row)
    ordered_columns = ["split", "total_segments"] + [label_names[idx] for idx in sorted(label_names)]
    return pd.DataFrame(rows, columns=ordered_columns)


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def normalize_peak_amplitude(waveform: torch.Tensor) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak > 1e-8:
        waveform = waveform / peak
    return waveform.clamp(-1.0, 1.0)


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_ms: int,
    hop_ms: int,
    energy_threshold: float,
) -> torch.Tensor:
    frame_length = max(1, int(sample_rate * frame_ms / 1000))
    hop_length = max(1, int(sample_rate * hop_ms / 1000))
    if waveform.numel() < frame_length:
        return waveform
    frames = waveform.unfold(0, frame_length, hop_length)
    energies = frames.pow(2).mean(dim=-1)
    mask = energies > energy_threshold
    if not mask.any():
        return waveform
    active = mask.nonzero(as_tuple=False).squeeze(-1)
    start = int(active[0]) * hop_length
    end = int(active[-1]) * hop_length + frame_length
    return waveform[start:min(end, waveform.size(0))]


def median_filter_1d(waveform: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel < 3 or kernel % 2 == 0:
        return waveform
    if waveform.numel() < kernel:
        return waveform
    pad = kernel // 2
    # torch.nn.functional.pad with reflect mode expects an input of at least 3 dimensions
    # for 1D padding (N, C, L). Reshape accordingly to avoid NotImplementedError.
    padded = F.pad(waveform.view(1, 1, -1), (pad, pad), mode="reflect").view(-1)
    windows = padded.unfold(0, kernel, 1)
    return windows.median(dim=-1).values


def preprocess_waveform(waveform: torch.Tensor, data_cfg: DataConfig) -> torch.Tensor:
    if waveform.dim() == 2:
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
    elif waveform.dim() == 0:
        waveform = waveform.view(1)
    if data_cfg.normalize_amplitude:
        waveform = normalize_peak_amplitude(waveform)
    if data_cfg.apply_silence_trim:
        waveform = trim_silence(
            waveform,
            data_cfg.sample_rate,
            data_cfg.silence_frame_ms,
            data_cfg.silence_hop_ms,
            data_cfg.silence_energy_threshold,
        )
    if data_cfg.apply_median_filter:
        waveform = median_filter_1d(waveform, data_cfg.median_filter_kernel)
    if waveform.numel() == 0:
        waveform = torch.zeros(int(data_cfg.sample_rate * 0.5), dtype=torch.float32)
    return waveform.reshape(-1).contiguous()


def safe_audio_info(path: str) -> Tuple[int, int]:
    try:
        info = torchaudio.info(path)
        return info.num_frames, info.sample_rate
    except (RuntimeError, OSError):
        with sf.SoundFile(path) as handle:
            return len(handle), handle.samplerate


def safe_audio_load(path: str) -> Tuple[torch.Tensor, int]:
    try:
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate
    except (RuntimeError, OSError):
        data, sample_rate = sf.read(path, always_2d=True)
        waveform = torch.from_numpy(data.T).float()
        return waveform, sample_rate


# ---------------------------------------------------------------------------
# Dataset and collate function
# ---------------------------------------------------------------------------

_feature_extractor: Optional[Wav2Vec2FeatureExtractor] = None


def get_feature_extractor(model_cfg: ModelConfig) -> Wav2Vec2FeatureExtractor:
    global _feature_extractor
    if _feature_extractor is None:
        try:
            _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_cfg.model_name,
                cache_dir=model_cfg.hf_cache_dir,
                local_files_only=model_cfg.local_files_only,
            )
        except Exception as exc:
            if not model_cfg.local_files_only and _is_hf_timeout_error(exc):
                _enable_hf_offline_mode(f"Failed to reach Hugging Face hub: {exc}.")
                model_cfg.local_files_only = True
                _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    model_cfg.model_name,
                    cache_dir=model_cfg.hf_cache_dir,
                    local_files_only=True,
                )
            else:
                raise
    return _feature_extractor


class AudioDataset(Dataset):
    def __init__(self, samples: List[Dict], data_cfg: DataConfig, split: str) -> None:
        self.data_cfg = data_cfg
        self.split = split
        self.segment_length = data_cfg.sample_rate * data_cfg.segment_duration
        self.apply_augmentation = data_cfg.apply_augmentation and split == "train"
        self.samples = samples
        self.class_counts = Counter(sample["label"] for sample in self.samples)
        self.subject_counts = Counter(sample["subject"] for sample in self.samples)
        weights: List[float] = []
        for sample in self.samples:
            class_w = 1.0 / self.class_counts[sample["label"]]
            subject_w = 1.0 / self.subject_counts[sample["subject"]]
            weights.append(class_w * subject_w)
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.mean() if len(weights) else weights
        self.sample_weights = torch.from_numpy(weights).double()
        self._cache_path: Optional[str] = None
        self._cache_waveform: Optional[torch.Tensor] = None

    @classmethod
    def build_samples(cls, data: Dict[str, List], data_cfg: DataConfig) -> List[Dict]:
        segment_length = data_cfg.sample_rate * data_cfg.segment_duration
        hop_length = max(1, int(segment_length * (1 - data_cfg.overlap_ratio)))
        samples: List[Dict] = []
        subject_counts = Counter()
        for path, label, subject in tqdm(
            list(zip(data["paths"], data["labels"], data["subjects"])),
            desc="Indexing audio",
            total=len(data["paths"]),
        ):
            try:
                num_frames, _ = safe_audio_info(path)
            except Exception as exc:
                logger.warning("Skipping %s | %s", path, exc)
                continue
            if num_frames <= 0:
                continue
            max_segments = data_cfg.max_segments_per_subject
            if num_frames <= segment_length:
                samples.append({"path": path, "label": label, "subject": subject, "offset": 0})
                subject_counts[subject] += 1
                continue
            added = 0
            for start in range(0, num_frames - segment_length + 1, hop_length):
                if max_segments is not None and subject_counts[subject] >= max_segments:
                    break
                samples.append({"path": path, "label": label, "subject": subject, "offset": start})
                subject_counts[subject] += 1
                added += 1
            if added == 0 and (max_segments is None or subject_counts[subject] < max_segments):
                samples.append({"path": path, "label": label, "subject": subject, "offset": 0})
                subject_counts[subject] += 1
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_waveform(self, path: str) -> torch.Tensor:
        if path != self._cache_path:
            waveform, sample_rate = safe_audio_load(path)
            if sample_rate != self.data_cfg.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.data_cfg.sample_rate)
            waveform = preprocess_waveform(waveform.squeeze(0), self.data_cfg)
            self._cache_path = path
            self._cache_waveform = waveform.contiguous()
        return self._cache_waveform.clone()

    def _crop_segment(self, waveform: torch.Tensor, offset: int) -> Tuple[torch.Tensor, int]:
        target = self.segment_length
        if waveform.size(0) <= target:
            length = waveform.size(0)
            padded = F.pad(waveform, (0, target - waveform.size(0)))
            return padded, length
        offset = int(min(max(offset, 0), waveform.size(0) - target))
        segment = waveform[offset:offset + target]
        return segment, target

    def _time_stretch(self, segment: torch.Tensor) -> torch.Tensor:
        min_rate, max_rate = self.data_cfg.time_stretch_range
        if max_rate <= 0 or min_rate <= 0:
            return segment
        rate = random.uniform(min_rate, max_rate)
        if abs(rate - 1.0) < 1e-2:
            return segment
        new_sr = max(1, int(self.data_cfg.sample_rate * rate))
        stretched = torchaudio.functional.resample(
            segment.unsqueeze(0),
            self.data_cfg.sample_rate,
            new_sr,
        ).squeeze(0)
        target = segment.size(0)
        if stretched.size(0) > target:
            start = random.randint(0, stretched.size(0) - target)
            stretched = stretched[start:start + target]
        elif stretched.size(0) < target:
            stretched = F.pad(stretched, (0, target - stretched.size(0)))
        return stretched

    def _time_shift(self, segment: torch.Tensor) -> torch.Tensor:
        max_ratio = self.data_cfg.time_shift_max_ratio
        if max_ratio <= 0:
            return segment
        max_shift = int(segment.size(0) * max_ratio)
        if max_shift <= 0:
            return segment
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return segment
        return torch.roll(segment, shifts=shift)

    def _time_dropout(self, segment: torch.Tensor) -> torch.Tensor:
        max_ratio = self.data_cfg.time_dropout_max_ratio
        if max_ratio <= 0:
            return segment
        max_span = int(segment.size(0) * max_ratio)
        if max_span <= 0:
            return segment
        span = random.randint(1, max_span)
        start = random.randint(0, max(0, segment.size(0) - span))
        dropped = segment.clone()
        dropped[start:start + span] = 0.0
        return dropped

    def _augment(self, segment: torch.Tensor) -> torch.Tensor:
        if not self.apply_augmentation or random.random() > self.data_cfg.augmentation_prob:
            return segment
        if random.random() < self.data_cfg.time_stretch_prob:
            segment = self._time_stretch(segment)
        if random.random() < self.data_cfg.time_shift_prob:
            segment = self._time_shift(segment)
        if random.random() < self.data_cfg.time_dropout_prob:
            segment = self._time_dropout(segment)
        if random.random() < 0.6:
            std_min, std_max = self.data_cfg.noise_std_range
            noise_std = random.uniform(std_min, std_max)
            segment = segment + torch.randn_like(segment) * noise_std
        if random.random() < 0.5:
            gain_min, gain_max = self.data_cfg.gain_range
            gain = random.uniform(gain_min, gain_max)
            segment = segment * gain
        if random.random() < 0.3:
            cutoff = random.uniform(200.0, min(6000.0, 0.45 * self.data_cfg.sample_rate))
            segment = torchaudio.functional.lowpass_biquad(
                segment.unsqueeze(0), self.data_cfg.sample_rate, cutoff
            ).squeeze(0)
        if random.random() < 0.3:
            cutoff = random.uniform(50.0, min(1500.0, 0.45 * self.data_cfg.sample_rate))
            segment = torchaudio.functional.highpass_biquad(
                segment.unsqueeze(0), self.data_cfg.sample_rate, cutoff
            ).squeeze(0)
        segment = segment.clamp(-1.0, 1.0)
        return segment

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, int]:
        sample = self.samples[index]
        waveform = self._load_waveform(sample["path"])
        segment, length = self._crop_segment(waveform, sample["offset"])
        if self.apply_augmentation:
            segment = self._augment(segment)
        return segment.float(), sample["label"], sample["subject"], length


# ---------------------------------------------------------------------------
# Subject-level evaluation utilities
# ---------------------------------------------------------------------------


@dataclass
class SubjectLevelData:
    subject_ids: List[str]
    labels: np.ndarray
    probabilities: np.ndarray


def aggregate_subject_predictions(
    labels: List[int],
    probabilities: List[float],
    subjects: List[str],
) -> SubjectLevelData:
    if not probabilities:
        return SubjectLevelData([], np.asarray([]), np.asarray([]))
    frame = pd.DataFrame({"subject": subjects, "label": labels, "prob": probabilities})
    grouped = frame.groupby("subject", sort=False)
    subj_probs = grouped["prob"].mean()
    subj_labels = grouped["label"].first()
    return SubjectLevelData(
        subject_ids=list(subj_probs.index),
        labels=subj_labels.to_numpy(dtype=np.int64, copy=True),
        probabilities=subj_probs.to_numpy(dtype=np.float64, copy=True),
    )


def compute_threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    preds = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    acc = accuracy_score(labels, preds) if labels.size else 0.0
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    youden_j = sensitivity + specificity - 1.0
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "npv": float(npv),
        "balanced_accuracy": float(balanced_acc),
        "f1": float(f1),
        "mcc": float(mcc),
        "youden_j": float(youden_j),
        "confusion_matrix": cm,
    }


def scan_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    beta: float = 2.0,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for thr in thresholds:
        metrics = compute_threshold_metrics(labels, probabilities, float(thr))
        tn, fp, fn, tp = metrics["confusion_matrix"].ravel()
        beta_sq = beta ** 2
        fbeta = (1 + beta_sq) * tp
        denom = (1 + beta_sq) * tp + beta_sq * fn + fp
        fbeta = fbeta / denom if denom else 0.0
        record = {
            "threshold": float(thr),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "npv": metrics["npv"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1": metrics["f1"],
            "mcc": metrics["mcc"],
            "youden_j": metrics["youden_j"],
            f"f{beta:g}": fbeta,
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def find_best_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    beta: float = 2.0,
) -> Tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.01, 0.99, 99)
    scan_df = scan_thresholds(labels, probabilities, thresholds, beta=beta)
    score_col = f"f{beta:g}"
    if scan_df.empty:
        return 0.5, scan_df
    best_row = scan_df.loc[scan_df[score_col].idxmax()]
    return float(best_row["threshold"]), scan_df


def bootstrap_metric_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    random_state: int = 24,
) -> Tuple[float, float]:
    if labels.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    estimates: List[float] = []
    n = labels.size
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_labels = labels[indices]
        if len(np.unique(sample_labels)) < 2:
            continue
        sample_probs = probabilities[indices]
        estimates.append(float(metric_fn(sample_labels, sample_probs)))
    if not estimates:
        return float("nan"), float("nan")
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def bootstrap_curve_band(
    labels: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray,
    curve_type: str,
    n_bootstrap: int = 300,
    random_state: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    interpolated: List[np.ndarray] = []
    n = labels.size
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_labels = labels[indices]
        if len(np.unique(sample_labels)) < 2:
            continue
        sample_probs = probabilities[indices]
        if curve_type == "roc":
            fpr, tpr, _ = roc_curve(sample_labels, sample_probs)
            interp_vals = np.interp(grid, fpr, tpr, left=0.0, right=1.0)
        else:
            precision, recall, _ = precision_recall_curve(sample_labels, sample_probs)
            # recall is sorted descending; reverse for interpolation
            recall = recall[::-1]
            precision = precision[::-1]
            interp_vals = np.interp(grid, recall, precision, left=precision[0], right=precision[-1])
        interpolated.append(interp_vals)
    if not interpolated:
        return np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    stacked = np.vstack(interpolated)
    lower = np.percentile(stacked, 2.5, axis=0)
    upper = np.percentile(stacked, 97.5, axis=0)
    return lower, upper


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = labels.size
    if total == 0:
        return float("nan")
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (probabilities >= bins[i]) & (probabilities < bins[i + 1])
        else:
            mask = (probabilities >= bins[i]) & (probabilities <= bins[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_prob = probabilities[mask].mean()
        avg_label = labels[mask].mean()
        ece += (count / total) * abs(avg_prob - avg_label)
    return float(ece)


def decision_curve_analysis(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = labels.size
    prevalence = labels.mean() if n else 0.0
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    treat_none = np.zeros_like(thresholds)
    net_benefit = []
    for thr in thresholds:
        preds = (probabilities >= thr).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        nb = (tp / n) - (fp / n) * (thr / (1 - thr)) if n else 0.0
        net_benefit.append(nb)
    return np.asarray(net_benefit, dtype=np.float64), treat_all, treat_none


def plot_roc_curve_with_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: str,
    positive_label: str,
    random_state: int = 24,
) -> float:
    if labels.size == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = roc_auc_score(labels, probabilities) if len(np.unique(labels)) > 1 else float("nan")
    grid = np.linspace(0.0, 1.0, 101)
    lower, upper = bootstrap_curve_band(labels, probabilities, grid, "roc", random_state=random_state)
    auc_ci = bootstrap_metric_ci(labels, probabilities, roc_auc_score, random_state=random_state)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})", color="C0")
    plt.fill_between(grid, lower, upper, color="C0", alpha=0.2, label="95% CI")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    plt.text(
        0.6,
        0.1,
        f"AUC 95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Positive Class={positive_label})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return roc_auc


def plot_pr_curve_with_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: str,
    positive_label: str,
    random_state: int = 24,
) -> float:
    if labels.size == 0:
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    pr_auc = average_precision_score(labels, probabilities) if len(np.unique(labels)) > 1 else float("nan")
    grid = np.linspace(0.0, 1.0, 101)
    lower, upper = bootstrap_curve_band(labels, probabilities, grid, "pr", random_state=random_state)
    ap_ci = bootstrap_metric_ci(labels, probabilities, average_precision_score, random_state=random_state)
    positive_rate = labels.mean() if labels.size else 0.0

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR (AP={pr_auc:.3f})", color="C1")
    plt.fill_between(grid, lower, upper, color="C1", alpha=0.2, label="95% CI")
    plt.hlines(positive_rate, 0, 1, colors="grey", linestyles="--", label=f"Baseline={positive_rate:.3f}")
    plt.text(
        0.55,
        max(positive_rate + 0.02, 0.02),
        f"AP 95% CI: [{ap_ci[0]:.3f}, {ap_ci[1]:.3f}]",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Positive Class={positive_label})")
    plt.ylim(0, 1.05)
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return pr_auc


def plot_threshold_diagnostics(
    scan_df: pd.DataFrame,
    best_threshold: float,
    output_path: str,
    beta: float = 2.0,
) -> None:
    if scan_df.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.plot(scan_df["threshold"], scan_df["recall"], label="Sensitivity (TPR)")
    plt.plot(scan_df["threshold"], scan_df["specificity"], label="Specificity (TNR)")
    plt.plot(scan_df["threshold"], scan_df["precision"], label="PPV")
    plt.plot(scan_df["threshold"], scan_df["npv"], label="NPV")
    plt.plot(scan_df["threshold"], scan_df["balanced_accuracy"], label="Balanced Acc")
    plt.plot(scan_df["threshold"], scan_df["f1"], label="F1")
    plt.plot(scan_df["threshold"], scan_df["mcc"], label="MCC")
    score_col = f"f{beta:g}"
    if score_col in scan_df.columns:
        plt.plot(scan_df["threshold"], scan_df[score_col], label=f"F{beta}")
    plt.axvline(best_threshold, color="black", linestyle="--", label=f"Best thr={best_threshold:.2f}")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold-dependent Metrics")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_sensitivity_specificity_bar(
    metrics: Dict[str, float],
    output_path: str,
) -> None:
    values = {
        "Sensitivity": metrics.get("recall", 0.0),
        "Specificity": metrics.get("specificity", 0.0),
        "PPV": metrics.get("precision", 0.0),
        "NPV": metrics.get("npv", 0.0),
    }
    plt.figure(figsize=(6, 5))
    bars = plt.bar(values.keys(), values.values(), color=["C0", "C1", "C2", "C3"])
    plt.ylim(0, 1.05)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.2f}", ha="center")
    plt.ylabel("Value")
    plt.title("Best-threshold Performance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrices(
    cm: np.ndarray,
    output_path: str,
    normalize: bool = False,
    cmap: str = "Blues",
) -> None:
    if cm.size != 4:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    matrices = [cm, normalized]
    titles = ["Confusion Matrix", "Normalized Confusion Matrix"]
    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["HC", "MDD"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["HC", "MDD"])
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.2f}" if title.startswith("Normalized") else f"{int(value)}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration_reliability(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: str,
    n_bins: int = 10,
) -> Tuple[float, float]:
    if labels.size == 0:
        return float("nan"), float("nan")
    prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(labels, probabilities) if len(np.unique(labels)) > 1 else float("nan")
    ece = expected_calibration_error(labels, probabilities, n_bins=n_bins)
    fig, (ax_cal, ax_hist) = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax_cal.plot(prob_pred, prob_true, marker="o", label="Model")
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Ideal")
    ax_cal.set_ylabel("Empirical Positive Rate")
    ax_cal.set_title("Reliability Diagram")
    ax_cal.legend()
    ax_cal.grid(True, alpha=0.3)
    ax_cal.text(
        0.05,
        0.8,
        f"Brier={brier:.3f}\nECE={ece:.3f}",
        transform=ax_cal.transAxes,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    ax_hist.hist(probabilities, bins=n_bins, range=(0, 1), color="C0", alpha=0.7)
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Prediction Probability Histogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return brier, ece


def plot_decision_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: str,
) -> None:
    thresholds = np.linspace(0.01, 0.99, 99)
    net_benefit, treat_all, treat_none = decision_curve_analysis(labels, probabilities, thresholds)
    plt.figure(figsize=(7, 6))
    plt.plot(thresholds, net_benefit, label="Model", color="C0")
    plt.plot(thresholds, treat_all, label="Treat-all", linestyle="--", color="C1")
    plt.plot(thresholds, treat_none, label="Treat-none", linestyle=":", color="C2")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_probability_distributions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: str,
) -> None:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    thresholds = np.linspace(0.0, 1.0, 200)
    pos_cdf = np.array([(positives <= t).mean() if positives.size else 0.0 for t in thresholds])
    neg_cdf = np.array([(negatives <= t).mean() if negatives.size else 0.0 for t in thresholds])
    ks_stat = np.max(np.abs(pos_cdf - neg_cdf)) if thresholds.size else float("nan")
    fig, (ax_hist, ax_ks) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax_hist.hist(negatives, bins=30, alpha=0.6, label="HC", color="C0", density=True)
    ax_hist.hist(positives, bins=30, alpha=0.6, label="MDD", color="C1", density=True)
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Predicted Probability Distributions")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    ax_ks.plot(thresholds, pos_cdf, label="CDF(MDD)")
    ax_ks.plot(thresholds, neg_cdf, label="CDF(HC)")
    ax_ks.fill_between(thresholds, pos_cdf, neg_cdf, color="C2", alpha=0.2)
    ax_ks.set_xlabel("Threshold")
    ax_ks.set_ylabel("CDF")
    ax_ks.set_title(f"KS Curve (KS={ks_stat:.3f})")
    ax_ks.legend()
    ax_ks.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_learning_curves(
    epochs: List[int],
    train_losses: List[float],
    val_losses: List[float],
    val_auc: List[float],
    val_pr_auc: List[float],
    val_f1: List[float],
    val_mcc: List[float],
    val_balanced_acc: List[float],
    output_path: str,
) -> None:
    if not epochs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o")
    axes[0].plot(epochs, val_losses, label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_auc, label="Val ROC-AUC", marker="o")
    axes[1].plot(epochs, val_pr_auc, label="Val PR-AUC", marker="s")
    axes[1].plot(epochs, val_f1, label="Val F1", marker="^")
    axes[1].plot(epochs, val_mcc, label="Val MCC", marker="v")
    axes[1].plot(epochs, val_balanced_acc, label="Val Balanced Acc", marker="d")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation Metrics over Epochs")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def collate_fn(
    batch: List[Tuple[torch.Tensor, int, str, int]],
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
):
    segments, labels, subjects, lengths = zip(*batch)
    extractor = get_feature_extractor(model_cfg)
    segments_np = [seg.numpy() for seg in segments]
    processed = extractor(
        segments_np,
        sampling_rate=data_cfg.sample_rate,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    attention_mask = processed.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(processed.input_values, dtype=torch.long)
    else:
        attention_mask = attention_mask.long()
    return (
        {
            "input_values": processed.input_values,
            "attention_mask": attention_mask,
            "sample_lengths": torch.tensor(lengths, dtype=torch.long),
        },
        torch.tensor(labels, dtype=torch.long),
        list(subjects),
    )


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, model_cfg: ModelConfig, num_classes: int) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        try:
            self.backbone = WavLMModel.from_pretrained(
                model_cfg.model_name,
                cache_dir=model_cfg.hf_cache_dir,
                local_files_only=model_cfg.local_files_only,
            )
        except Exception as exc:
            if not model_cfg.local_files_only and _is_hf_timeout_error(exc):
                _enable_hf_offline_mode(f"Failed to download {model_cfg.model_name}: {exc}.")
                model_cfg.local_files_only = True
                self.backbone = WavLMModel.from_pretrained(
                    model_cfg.model_name,
                    cache_dir=model_cfg.hf_cache_dir,
                    local_files_only=True,
                )
            else:
                raise
        hidden_size = self.backbone.config.hidden_size
        hidden_dim = model_cfg.classifier_hidden_dim if model_cfg.classifier_hidden_dim > 0 else hidden_size
        self.frame_dropout = nn.Dropout(model_cfg.frame_dropout)
        self.pre_classifier_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        if not trainable:
            return
        if hasattr(self.backbone, "feature_extractor"):
            for param in self.backbone.feature_extractor.parameters():
                param.requires_grad = False
        layers = getattr(self.backbone.encoder, "layers", None)
        if layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            n_layers = len(layers)
            target = self.model_cfg.unfreeze_last_n_layers or n_layers
            for layer in layers[-target:]:
                for param in layer.parameters():
                    param.requires_grad = True
        if hasattr(self.backbone, "layer_norm"):
            for param in self.backbone.layer_norm.parameters():
                param.requires_grad = True
        self.backbone.train()

    def head_parameters(self) -> List[nn.Parameter]:
        params = list(self.pre_classifier_norm.parameters()) + list(self.classifier.parameters())
        return [p for p in params if p.requires_grad]

    def backbone_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_mask = batch.get("attention_mask")
        outputs = self.backbone(
            batch["input_values"],
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_states = self.frame_dropout(outputs.last_hidden_state)
        if attention_mask is None:
            input_mask = torch.ones(
                batch["input_values"].size()[:2],
                device=hidden_states.device,
                dtype=torch.long,
            )
        else:
            input_mask = attention_mask.to(hidden_states.device)
        input_lengths = input_mask.sum(dim=1)
        if hasattr(self.backbone, "_get_feat_extract_output_lengths"):
            feat_lengths = self.backbone._get_feat_extract_output_lengths(input_lengths)
            feat_lengths = feat_lengths.to(hidden_states.device)
        else:
            conv_stride = getattr(self.backbone.config, "conv_stride", None)
            if conv_stride is None and hasattr(self.backbone, "feature_extractor"):
                conv_stride = [
                    layer.stride[0]
                    for layer in getattr(self.backbone.feature_extractor, "conv_layers", [])
                ]
            stride = int(np.prod(conv_stride)) if conv_stride else 1
            feat_lengths = torch.div(
                input_lengths + stride - 1,
                stride,
                rounding_mode="floor",
            ).to(hidden_states.device)
        max_len = hidden_states.size(1)
        frame_index = torch.arange(max_len, device=hidden_states.device).unsqueeze(0)
        frame_mask = frame_index < feat_lengths.unsqueeze(1)
        mask = frame_mask.unsqueeze(-1).type_as(hidden_states)
        masked = hidden_states * mask
        lengths = frame_mask.sum(dim=1, keepdim=True).clamp(min=1).type_as(hidden_states)
        pooled = masked.sum(dim=1) / lengths
        pooled = self.pre_classifier_norm(pooled)
        logits = self.classifier(self.dropout(pooled))
        return logits, pooled


# ---------------------------------------------------------------------------
# Training and evaluation utilities
# ---------------------------------------------------------------------------

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) if torch.is_tensor(tensor) else tensor for key, tensor in batch.items()}


def evaluate(
    model: Wav2Vec2Classifier,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    split_name: str,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_subjects: List[str] = []
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch, labels, subjects in tqdm(dataloader, desc=f"Evaluating[{split_name}]", leave=False):
            batch = move_batch_to_device(batch, device)
            labels = labels.to(device)
            with autocast(enabled=use_amp):
                logits, _ = model(batch)
                probs = torch.softmax(logits, dim=1)
            loss = F.cross_entropy(
                logits.float(),
                labels,
                reduction="sum",
                label_smoothing=CONFIG.train.label_smoothing,
            )
            total_loss += loss.item()
            total_samples += labels.size(0)
            pred_probs = probs.cpu()
            all_preds.extend(pred_probs.argmax(dim=1).tolist())
            all_probs.extend(pred_probs[:, 1].tolist())
            all_labels.extend(labels.cpu().tolist())
            all_subjects.extend(subjects)

    results: Dict[str, float] = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "loss": total_loss / max(total_samples, 1),
        "auc": float("nan"),
    }
    if not all_labels:
        results["confusion_matrix"] = None
        return results

    results["accuracy"] = accuracy_score(all_labels, all_preds)
    results["precision"] = precision_score(all_labels, all_preds, zero_division=0)
    results["recall"] = recall_score(all_labels, all_preds, zero_division=0)
    results["f1"] = f1_score(all_labels, all_preds, zero_division=0)
    try:
        if len(set(all_labels)) > 1:
            results["auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        results["auc"] = float("nan")
    results["confusion_matrix"] = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    logger.info(
        "[%s] acc=%.4f | precision=%.4f | recall=%.4f | F1=%.4f | AUC=%.4f | loss=%.4f",
        split_name,
        results["accuracy"],
        results["precision"],
        results["recall"],
        results["f1"],
        results["auc"],
        results["loss"],
    )
    return results, {
        "labels": np.asarray(all_labels, dtype=np.int64),
        "probabilities": np.asarray(all_probs, dtype=np.float64),
        "subjects": np.asarray(all_subjects, dtype=object),
    }


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    val_precisions: List[float],
    val_recalls: List[float],
    val_f1s: List[float],
) -> None:
    if not train_losses:
        return
    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs, val_accuracies, label="Val Acc")
    axes[1].plot(epochs, val_precisions, label="Val Precision")
    axes[1].plot(epochs, val_recalls, label="Val Recall")
    axes[1].plot(epochs, val_f1s, label="Val F1")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(CONFIG.train.curve_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info("Saved training curves to %s", CONFIG.train.curve_path)


def build_dataloader(
    dataset: AudioDataset,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    sampler: Optional[WeightedRandomSampler],
    shuffle: bool,
) -> DataLoader:
    def _collate(batch):
        return collate_fn(batch, data_cfg, model_cfg)

    return DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=train_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
        persistent_workers=train_cfg.persistent_workers and train_cfg.num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Training workflow
# ---------------------------------------------------------------------------

def determine_total_epochs(train_cfg: TrainConfig) -> int:
    return train_cfg.pretrained_freeze_epochs + train_cfg.pretrained_finetune_epochs


def main() -> None:
    setup_logging()
    logger.info("Starting experiment")
    if CONFIG.model.force_offline:
        _enable_hf_offline_mode("force_offline flag enabled.")
        CONFIG.model.local_files_only = True
    os.makedirs(CONFIG.model.hf_cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", CONFIG.model.hf_cache_dir)

    seed_everything(CONFIG.train.random_seed)

    file_paths, labels, subjects = load_metadata(CONFIG.data)
    logger.info("Total audio files available: %d", len(file_paths))
    full_data = {"paths": file_paths, "labels": labels, "subjects": subjects}
    all_samples = AudioDataset.build_samples(full_data, CONFIG.data)
    logger.info("Constructed %d segments from %d audio files", len(all_samples), len(file_paths))

    train_samples, val_samples, test_samples = split_segments(all_samples, CONFIG.train)
    logger.info(
        "Segment split sizes | train=%d | val=%d | test=%d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )

    split_summary_df = summarize_split_counts(
        {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }
    )
    if not split_summary_df.empty:
        display_df = split_summary_df.set_index("split")
        try:
            from IPython.display import display  # type: ignore

            display(display_df)
        except Exception:
            logger.info("Segment label distribution by split:\n%s", display_df.to_string())
        else:
            logger.info("Segment label distribution by split:\n%s", display_df.to_string())
    else:
        logger.info("No segment samples available to summarise label distribution.")

    train_dataset = AudioDataset(train_samples, CONFIG.data, split="train")
    val_dataset = AudioDataset(val_samples, CONFIG.data, split="val")
    test_dataset = AudioDataset(test_samples, CONFIG.data, split="test")
    if not len(train_dataset):
        raise RuntimeError("Training dataset is empty.")

    sampler = None
    if train_dataset.sample_weights.numel() > 0:
        sampler = WeightedRandomSampler(
            train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=False,
        )

    train_loader = build_dataloader(train_dataset, CONFIG.data, CONFIG.model, CONFIG.train, sampler, shuffle=True)
    val_loader = build_dataloader(val_dataset, CONFIG.data, CONFIG.model, CONFIG.train, sampler=None, shuffle=False)
    test_loader = build_dataloader(test_dataset, CONFIG.data, CONFIG.model, CONFIG.train, sampler=None, shuffle=False)

    num_classes = len(train_dataset.class_counts) if train_dataset.class_counts else len(set(labels))
    model = Wav2Vec2Classifier(CONFIG.model, num_classes).to(CONFIG.device)

    optimizer_groups = [
        {"params": model.head_parameters(), "lr": CONFIG.train.head_learning_rate},
    ]
    optimizer = torch.optim.Adam(optimizer_groups, weight_decay=CONFIG.train.weight_decay)
    total_epochs = determine_total_epochs(CONFIG.train)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=CONFIG.train.scheduler_eta_min,
    )
    scaler = GradScaler(enabled=CONFIG.train.use_amp and CONFIG.device.type == "cuda")

    best_val_metric = float("-inf")
    best_threshold = 0.5
    backbone_params_added = False
    epochs_without_improvement = 0

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_subject_aucs: List[float] = []
    val_subject_pr_aucs: List[float] = []
    val_subject_f1s: List[float] = []
    val_subject_mccs: List[float] = []
    val_subject_balanced_accs: List[float] = []
    epochs_tracked: List[int] = []
    history_records: List[Dict[str, float]] = []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(1, total_epochs + 1):
        if (
            not backbone_params_added
            and epoch > CONFIG.train.pretrained_freeze_epochs
        ):
            model.set_backbone_trainable(True)
            backbone_params = model.backbone_parameters()
            if backbone_params:
                optimizer.add_param_group({"params": backbone_params, "lr": CONFIG.train.backbone_learning_rate})
                scheduler.base_lrs.append(CONFIG.train.backbone_learning_rate)
                backbone_params_added = True
                logger.info(
                    "Unfroze last %d transformer layers with lr=%.2e",
                    CONFIG.model.unfreeze_last_n_layers,
                    CONFIG.train.backbone_learning_rate,
                )

        model.train()
        total_loss = 0.0
        correct = 0
        total_examples = 0
        train_preds: List[int] = []
        train_labels: List[int] = []
        amp_enabled = CONFIG.train.use_amp
        nan_warnings = 0
        suppression_logged = False
        lr_scaled = False

        progress = tqdm(train_loader, desc="Training", leave=False, dynamic_ncols=True)
        for batch_data, labels, _ in progress:
            labels = labels.to(CONFIG.device)
            batch_data = move_batch_to_device(batch_data, CONFIG.device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                logits, _ = model(batch_data)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=CONFIG.train.label_smoothing,
                )
            if not torch.isfinite(loss):
                nan_warnings += 1
                if nan_warnings <= CONFIG.train.max_nan_warnings:
                    logger.warning("Non-finite loss encountered (value=%s). Skipping batch.", loss.item())
                elif not suppression_logged:
                    logger.warning(
                        "Additional non-finite loss warnings suppressed for this epoch (already %d occurrences).",
                        nan_warnings,
                    )
                    suppression_logged = True
                optimizer.zero_grad(set_to_none=True)
                if CONFIG.train.nan_lr_scale < 1.0:
                    for group in optimizer.param_groups:
                        new_lr = max(CONFIG.train.scheduler_eta_min, group["lr"] * CONFIG.train.nan_lr_scale)
                        if new_lr < group["lr"]:
                            group["lr"] = new_lr
                            lr_scaled = True
                if lr_scaled and nan_warnings == 1:
                    logger.info(
                        "Reduced learning rates by factor %.2f to mitigate instabilities.",
                        CONFIG.train.nan_lr_scale,
                    )
                if CONFIG.train.disable_amp_on_nan and amp_enabled:
                    amp_enabled = False
                    logger.warning("Disabling AMP for the remainder of this epoch due to non-finite loss.")
                continue

            scaler.scale(loss).backward()
            if CONFIG.train.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), CONFIG.train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            preds = logits.detach().argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_examples += labels.size(0)
            total_loss += loss.item()
            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / max(total_examples, 1):.2f}%",
            )

        avg_loss = total_loss / max(len(train_loader), 1)
        train_acc = correct / total_examples if total_examples else 0.0
        train_f1 = f1_score(train_labels, train_preds, zero_division=0) if train_labels else 0.0
        logger.info(
            "[Train-epoch%02d/%02d] loss=%.4f | acc=%.4f | F1=%.4f",
            epoch,
            total_epochs,
            avg_loss,
            train_acc,
            train_f1,
        )

        train_losses.append(avg_loss)
        epochs_tracked.append(epoch)

        val_result, val_raw = evaluate(
            model,
            val_loader,
            CONFIG.device,
            use_amp=False,
            split_name=f"Val-epoch{epoch:02d}/{total_epochs}",
        )
        val_losses.append(val_result.get("loss", 0.0))

        subject_data = aggregate_subject_predictions(
            val_raw["labels"].tolist(),
            val_raw["probabilities"].tolist(),
            val_raw["subjects"].tolist(),
        )

        if subject_data.labels.size > 0:
            if len(np.unique(subject_data.labels)) > 1:
                subject_auc = roc_auc_score(subject_data.labels, subject_data.probabilities)
                subject_pr_auc = average_precision_score(subject_data.labels, subject_data.probabilities)
            else:
                subject_auc = float("nan")
                subject_pr_auc = float("nan")
        else:
            subject_auc = float("nan")
            subject_pr_auc = float("nan")

        best_thr_candidate, threshold_scan = find_best_threshold(subject_data.labels, subject_data.probabilities, beta=2.0)
        threshold_metrics = compute_threshold_metrics(subject_data.labels, subject_data.probabilities, best_thr_candidate)
        subject_f1 = threshold_metrics["f1"]
        subject_mcc = threshold_metrics["mcc"]
        subject_balanced_acc = threshold_metrics["balanced_accuracy"]

        val_subject_aucs.append(subject_auc)
        val_subject_pr_aucs.append(subject_pr_auc)
        val_subject_f1s.append(subject_f1)
        val_subject_mccs.append(subject_mcc)
        val_subject_balanced_accs.append(subject_balanced_acc)

        logger.info(
            "[Val-epoch%02d/%02d] loss=%.4f | subj AUC=%.4f | subj PR-AUC=%.4f | subj F1=%.4f | subj MCC=%.4f | subj BalAcc=%.4f",
            epoch,
            total_epochs,
            val_result.get("loss", 0.0),
            subject_auc,
            subject_pr_auc,
            subject_f1,
            subject_mcc,
            subject_balanced_acc,
        )

        history_records.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_result.get("loss", 0.0),
                "val_subject_auc": subject_auc,
                "val_subject_pr_auc": subject_pr_auc,
                "val_subject_f1": subject_f1,
                "val_subject_mcc": subject_mcc,
                "val_subject_balanced_acc": subject_balanced_acc,
                "val_best_threshold": best_thr_candidate,
            }
        )

        if subject_pr_auc > best_val_metric:
            best_val_metric = subject_pr_auc
            best_threshold = best_thr_candidate
            torch.save(model.state_dict(), CONFIG.train.best_model_path)
            logger.info(
                "Saved best model (epoch %d, val PR-AUC=%.4f, thr=%.2f) -> %s",
                epoch,
                best_val_metric,
                best_threshold,
                CONFIG.train.best_model_path,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            logger.info(
                "Validation PR-AUC did not improve for %d epoch(s)",
                epochs_without_improvement,
            )
            if epochs_without_improvement >= 5:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    epochs_without_improvement,
                )
                break

        scheduler.step()

    if history_records:
        history_df = pd.DataFrame(history_records)
        history_path = os.path.join(CONFIG.train.log_dir, f"training_history_{run_id}.csv")
        history_df.to_csv(history_path, index=False)
        logger.info("Saved training history to %s", history_path)

    if CONFIG.train.plot_training_curves and train_losses:
        plot_learning_curves(
            epochs_tracked,
            train_losses,
            val_losses,
            val_subject_aucs,
            val_subject_pr_aucs,
            val_subject_f1s,
            val_subject_mccs,
            val_subject_balanced_accs,
            CONFIG.train.curve_path,
        )

    logger.info("Evaluating on test set")
    if os.path.exists(CONFIG.train.best_model_path):
        model.load_state_dict(torch.load(CONFIG.train.best_model_path, map_location=CONFIG.device))
    test_result, test_raw = evaluate(
        model,
        test_loader,
        CONFIG.device,
        use_amp=False,
        split_name="Test",
    )
    logger.info(
        "[Test-Segment] acc=%.4f | precision=%.4f | recall=%.4f | F1=%.4f | AUC=%.4f | loss=%.4f",
        test_result.get("accuracy", 0.0),
        test_result.get("precision", 0.0),
        test_result.get("recall", 0.0),
        test_result.get("f1", 0.0),
        test_result.get("auc", float("nan")),
        test_result.get("loss", 0.0),
    )

    subject_test = aggregate_subject_predictions(
        test_raw["labels"].tolist(),
        test_raw["probabilities"].tolist(),
        test_raw["subjects"].tolist(),
    )
    test_metrics = compute_threshold_metrics(subject_test.labels, subject_test.probabilities, best_threshold)
    test_auc = roc_auc_score(subject_test.labels, subject_test.probabilities) if len(np.unique(subject_test.labels)) > 1 else float("nan")
    test_pr_auc = average_precision_score(subject_test.labels, subject_test.probabilities) if len(np.unique(subject_test.labels)) > 1 else float("nan")
    logger.info(
        "[Test-Subject] thr=%.2f | AUC=%.4f | PR-AUC=%.4f | Sens=%.4f | Spec=%.4f | F1=%.4f | MCC=%.4f | BalAcc=%.4f",
        best_threshold,
        test_auc,
        test_pr_auc,
        test_metrics.get("recall", 0.0),
        test_metrics.get("specificity", 0.0),
        test_metrics.get("f1", 0.0),
        test_metrics.get("mcc", 0.0),
        test_metrics.get("balanced_accuracy", 0.0),
    )

    output_prefix = os.path.join(CONFIG.train.log_dir, "test_subject")
    os.makedirs(CONFIG.train.log_dir, exist_ok=True)
    roc_path = f"{output_prefix}_roc.png"
    pr_path = f"{output_prefix}_pr.png"
    threshold_path = f"{output_prefix}_threshold.png"
    bar_path = f"{output_prefix}_sens_spec.png"
    cm_path = f"{output_prefix}_confusion.png"
    calib_path = f"{output_prefix}_calibration.png"
    dca_path = f"{output_prefix}_dca.png"
    dist_path = f"{output_prefix}_probability.png"

    plot_roc_curve_with_ci(subject_test.labels, subject_test.probabilities, roc_path, "MDD")
    plot_pr_curve_with_ci(subject_test.labels, subject_test.probabilities, pr_path, "MDD")
    threshold_scan = scan_thresholds(subject_test.labels, subject_test.probabilities, np.linspace(0.01, 0.99, 99), beta=2.0)
    plot_threshold_diagnostics(threshold_scan, best_threshold, threshold_path, beta=2.0)
    plot_sensitivity_specificity_bar(test_metrics, bar_path)
    plot_confusion_matrices(test_metrics["confusion_matrix"], cm_path)
    brier, ece = plot_calibration_reliability(subject_test.labels, subject_test.probabilities, calib_path)
    plot_decision_curve(subject_test.labels, subject_test.probabilities, dca_path)
    plot_probability_distributions(subject_test.labels, subject_test.probabilities, dist_path)

    logger.info("Calibration metrics | Brier=%.4f | ECE=%.4f", brier, ece)


if __name__ == "__main__":
    main()
