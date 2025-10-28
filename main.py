#!/usr/bin/env python3
"""Training script for wav2vec2.0 on the MODMA-style dataset.

This module implements a lightweight pipeline that
    * loads metadata from the provided Excel file,
    * splits the data by subject into train/val/test sets,
    * chunks long recordings into fixed-length segments,
    * trains a classifier on top of wav2vec2 features,
    * evaluates both segment-level and subject-level metrics.
"""

import logging
import math
import os
import random
import warnings
from collections import Counter, defaultdict
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

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
    segment_duration: int = 3
    overlap_ratio: float = 0.0
    max_segments_per_subject: Optional[int] = None
    normalize_amplitude: bool = True
    apply_silence_trim: bool = True
    silence_frame_ms: int = 25
    silence_hop_ms: int = 10
    silence_energy_threshold: float = 1e-4
    apply_median_filter: bool = True
    median_filter_kernel: int = 5


@dataclass
class ModelConfig:
    model_name: str = "facebook/wav2vec2-base"
    hf_cache_dir: str = "./hf_cache"
    local_files_only: bool = False
    force_offline: bool = False
    dropout: float = 0.1
    unfreeze_last_n_layers: int = 2

    @property
    def tag(self) -> str:
        return _safe_model_tag(self.model_name)


@dataclass
class TrainConfig:
    batch_size: int = 16
    num_workers: int = 4
    persistent_workers: bool = True
    head_learning_rate: float = 1e-4
    backbone_learning_rate: float = 5e-6
    weight_decay: float = 1e-5
    scheduler_eta_min: float = 1e-6
    max_grad_norm: float = 1.0
    use_amp: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 24
    resume_from_best: bool = False
    best_model_path: str = "best_model.pt"
    log_dir: str = "logs_wav2vec2"
    curve_path: str = "curves_wav2vec2.png"
    val_selection_metric: str = "subject"
    log_eval_details: bool = True
    plot_training_curves: bool = True
    max_nan_warnings: int = 3
    nan_lr_scale: float = 0.5
    disable_amp_on_nan: bool = True
    pretrained_freeze_epochs: int = 15
    pretrained_finetune_epochs: int = 10


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = ExperimentConfig()
CONFIG.train.best_model_path = f"best_model_{CONFIG.model.tag}.pt"
CONFIG.train.log_dir = f"logs_{CONFIG.model.tag}"
CONFIG.train.curve_path = f"training_curves_by_{CONFIG.model.tag}.png"


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


def split_by_subject(
    file_paths: List[str],
    labels: List[int],
    subjects: List[str],
    train_cfg: TrainConfig,
) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]:
    subject_to_label: Dict[str, int] = {}
    for subject, label in zip(subjects, labels):
        subject_to_label.setdefault(subject, label)

    rng = random.Random(train_cfg.random_seed)
    hc_subjects = [s for s, lbl in subject_to_label.items() if lbl == 0]
    mdd_subjects = [s for s, lbl in subject_to_label.items() if lbl == 1]
    rng.shuffle(hc_subjects)
    rng.shuffle(mdd_subjects)

    def _split(bucket: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n_train = int(len(bucket) * train_cfg.train_ratio)
        n_val = int(len(bucket) * train_cfg.val_ratio)
        train = bucket[:n_train]
        val = bucket[n_train:n_train + n_val]
        test = bucket[n_train + n_val:]
        return train, val, test

    train_subjects, val_subjects, test_subjects = set(), set(), set()
    for subset in (_split(hc_subjects), _split(mdd_subjects)):
        train_subjects.update(subset[0])
        val_subjects.update(subset[1])
        test_subjects.update(subset[2])

    def _collect(target_subjects: set) -> Dict[str, List]:
        subset = {"paths": [], "labels": [], "subjects": []}
        for path, label, subject in zip(file_paths, labels, subjects):
            if subject in target_subjects:
                subset["paths"].append(path)
                subset["labels"].append(label)
                subset["subjects"].append(subject)
        return subset

    train_data = _collect(train_subjects)
    val_data = _collect(val_subjects)
    test_data = _collect(test_subjects)
    logger.info(
        "Split summary | train: %d subjects (%d files) | val: %d subjects (%d files) | test: %d subjects (%d files)",
        len(train_subjects), len(train_data["paths"]),
        len(val_subjects), len(val_data["paths"]),
        len(test_subjects), len(test_data["paths"]),
    )
    return train_data, val_data, test_data


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
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_cfg.model_name,
            cache_dir=model_cfg.hf_cache_dir,
            local_files_only=model_cfg.local_files_only,
        )
    return _feature_extractor


class AudioDataset(Dataset):
    def __init__(self, data: Dict[str, List], data_cfg: DataConfig, split: str) -> None:
        self.data_cfg = data_cfg
        self.split = split
        self.segment_length = data_cfg.sample_rate * data_cfg.segment_duration
        self.hop_length = max(1, int(self.segment_length * (1 - data_cfg.overlap_ratio)))
        self.samples: List[Dict] = []
        subject_counts = Counter()
        for path, label, subject in tqdm(
            list(zip(data["paths"], data["labels"], data["subjects"])),
            desc=f"Indexing audio [{split}]",
            total=len(data["paths"]),
        ):
            try:
                num_frames, _ = safe_audio_info(path)
            except Exception as exc:
                logger.warning("Skipping %s | %s", path, exc)
                continue
            if num_frames <= 0:
                continue
            max_segments = self.data_cfg.max_segments_per_subject
            if num_frames <= self.segment_length:
                self.samples.append({"path": path, "label": label, "subject": subject, "offset": 0})
                subject_counts[subject] += 1
                continue
            added = 0
            for start in range(0, num_frames - self.segment_length + 1, self.hop_length):
                if max_segments is not None and subject_counts[subject] >= max_segments:
                    break
                self.samples.append({"path": path, "label": label, "subject": subject, "offset": start})
                subject_counts[subject] += 1
                added += 1
            if added == 0 and (max_segments is None or subject_counts[subject] < max_segments):
                self.samples.append({"path": path, "label": label, "subject": subject, "offset": 0})
                subject_counts[subject] += 1
        self.class_counts = Counter(sample["label"] for sample in self.samples)
        self.subject_counts = Counter(sample["subject"] for sample in self.samples)
        weights = []
        for sample in self.samples:
            class_w = 1.0 / self.class_counts[sample["label"]]
            subject_w = 1.0 / self.subject_counts[sample["subject"]]
            weights.append(class_w * subject_w)
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.mean() if len(weights) else weights
        self.sample_weights = torch.from_numpy(weights).double()
        self._cache_path: Optional[str] = None
        self._cache_waveform: Optional[torch.Tensor] = None

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, int]:
        sample = self.samples[index]
        waveform = self._load_waveform(sample["path"])
        segment, length = self._crop_segment(waveform, sample["offset"])
        return segment.float(), sample["label"], sample["subject"], length


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
        self.backbone = Wav2Vec2Model.from_pretrained(
            model_cfg.model_name,
            cache_dir=model_cfg.hf_cache_dir,
            local_files_only=model_cfg.local_files_only,
        )
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
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
        return [p for p in list(self.dropout.parameters()) + list(self.classifier.parameters()) if p.requires_grad]

    def backbone_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_mask = batch.get("attention_mask")
        outputs = self.backbone(
            batch["input_values"],
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state
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
        logits = self.classifier(self.dropout(pooled))
        return logits, pooled


# ---------------------------------------------------------------------------
# Training and evaluation utilities
# ---------------------------------------------------------------------------

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) if torch.is_tensor(tensor) else tensor for key, tensor in batch.items()}


def compute_subject_metrics(
    logits: List[List[float]],
    labels: List[int],
    subjects: List[str],
) -> Tuple[float, Optional[np.ndarray]]:
    subject_logits: Dict[str, List[List[float]]] = defaultdict(list)
    subject_true: Dict[str, int] = {}
    for logit, label, subject in zip(logits, labels, subjects):
        subject_logits[subject].append(logit)
        subject_true[subject] = label
    if not subject_logits:
        return 0.0, None
    subject_preds: List[int] = []
    subject_labels: List[int] = []
    for subject, logit_list in subject_logits.items():
        mean_logits = torch.tensor(logit_list, dtype=torch.float32).mean(dim=0)
        pred = int(torch.argmax(mean_logits).item())
        subject_preds.append(pred)
        subject_labels.append(subject_true[subject])
    subject_acc = accuracy_score(subject_labels, subject_preds)
    cm = confusion_matrix(subject_labels, subject_preds, labels=[0, 1]) if subject_labels else None
    return subject_acc, cm


def evaluate(
    model: Wav2Vec2Classifier,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    split_name: str,
) -> Dict[str, float]:
    model.eval()
    all_logits: List[List[float]] = []
    all_labels: List[int] = []
    all_subjects: List[str] = []
    with torch.no_grad():
        for batch, labels, subjects in tqdm(dataloader, desc=f"Evaluating[{split_name}]", leave=False):
            batch = move_batch_to_device(batch, device)
            labels = labels.to(device)
            with autocast(enabled=use_amp):
                logits, _ = model(batch)
                probs = torch.softmax(logits, dim=1)
            all_logits.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_subjects.extend(subjects)

    results: Dict[str, float] = {
        "segment_acc": 0.0,
        "subject_acc": 0.0,
        "f1": 0.0,
        "auc": float("nan"),
    }
    if not all_labels:
        results["confusion_matrix"] = None
        results["subject_confusion_matrix"] = None
        return results

    preds = [int(np.argmax(logit)) for logit in all_logits]
    results["segment_acc"] = accuracy_score(all_labels, preds)
    results["f1"] = f1_score(all_labels, preds, zero_division=0)
    try:
        if len(set(all_labels)) > 1:
            results["auc"] = roc_auc_score(all_labels, [logit[1] for logit in all_logits])
    except ValueError:
        results["auc"] = float("nan")
    results["confusion_matrix"] = confusion_matrix(all_labels, preds, labels=[0, 1])

    subject_acc, subject_cm = compute_subject_metrics(all_logits, all_labels, all_subjects)
    results["subject_acc"] = subject_acc
    results["subject_confusion_matrix"] = subject_cm

    logger.info(
        "[%s] Segment acc=%.4f | Subject acc=%.4f | F1=%.4f | AUC=%.4f",
        split_name,
        results["segment_acc"],
        results["subject_acc"],
        results["f1"],
        results["auc"],
    )
    return results


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
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.makedirs(CONFIG.model.hf_cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", CONFIG.model.hf_cache_dir)

    seed_everything(CONFIG.train.random_seed)

    file_paths, labels, subjects = load_metadata(CONFIG.data)
    train_data, val_data, test_data = split_by_subject(file_paths, labels, subjects, CONFIG.train)

    train_dataset = AudioDataset(train_data, CONFIG.data, split="train")
    val_dataset = AudioDataset(val_data, CONFIG.data, split="val")
    test_dataset = AudioDataset(test_data, CONFIG.data, split="test")
    if not len(train_dataset):
        raise RuntimeError("Training dataset is empty.")

    sampler = None
    if train_dataset.sample_weights.numel() > 0:
        sampler = WeightedRandomSampler(
            train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )

    train_loader = build_dataloader(train_dataset, CONFIG.data, CONFIG.model, CONFIG.train, sampler, shuffle=True)
    val_loader = build_dataloader(val_dataset, CONFIG.data, CONFIG.model, CONFIG.train, sampler=None, shuffle=False)
    test_loader = build_dataloader(test_dataset, CONFIG.data, CONFIG.model, CONFIG.train, sampler=None, shuffle=False)

    num_classes = len(train_dataset.class_counts) if train_dataset.class_counts else len(set(labels))
    model = Wav2Vec2Classifier(CONFIG.model, num_classes).to(CONFIG.device)

    optimizer_groups = [
        {"params": model.head_parameters(), "lr": CONFIG.train.head_learning_rate},
    ]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=CONFIG.train.weight_decay)
    total_epochs = determine_total_epochs(CONFIG.train)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=CONFIG.train.scheduler_eta_min,
    )
    scaler = GradScaler(enabled=CONFIG.train.use_amp and CONFIG.device.type == "cuda")

    metric_field = "subject_acc" if CONFIG.train.val_selection_metric.lower() == "subject" else "segment_acc"
    best_val_score = float("-inf")
    backbone_params_added = False

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(1, total_epochs + 1):
        logger.info("Epoch %d/%d", epoch, total_epochs)
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
        total = 0
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
                loss = F.cross_entropy(logits, labels)
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

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / max(total, 1):.2f}%",
            )

        avg_loss = total_loss / max(len(train_loader), 1)
        accuracy = correct / total if total > 0 else 0.0
        logger.info("Train | loss=%.4f | acc=%.4f", avg_loss, accuracy)

        val_result = evaluate(
            model,
            val_loader,
            CONFIG.device,
            use_amp=False,
            split_name=f"val-epoch{epoch}",
        )
        current_score = val_result.get(metric_field, float("-inf"))
        if current_score > best_val_score:
            best_val_score = current_score
            torch.save(model.state_dict(), CONFIG.train.best_model_path)
            logger.info(
                "Saved best model (epoch %d, %s=%.4f) -> %s",
                epoch,
                metric_field,
                best_val_score,
                CONFIG.train.best_model_path,
            )

        scheduler.step()

    logger.info("Evaluating on test set")
    if os.path.exists(CONFIG.train.best_model_path):
        model.load_state_dict(torch.load(CONFIG.train.best_model_path, map_location=CONFIG.device))
    test_result = evaluate(
        model,
        test_loader,
        CONFIG.device,
        use_amp=False,
        split_name="test",
    )
    logger.info(
        "Test | segment_acc=%.4f | subject_acc=%.4f | f1=%.4f | auc=%.4f",
        test_result.get("segment_acc", 0.0),
        test_result.get("subject_acc", 0.0),
        test_result.get("f1", 0.0),
        test_result.get("auc", float("nan")),
    )
    logger.info("Confusion matrix:\n%s", test_result.get("confusion_matrix"))


if __name__ == "__main__":
    main()
