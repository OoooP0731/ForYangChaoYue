#!/usr/bin/env python3
#!/usr/bin/env python3
"""Subject-level training with WavLM-Large on the MODMA corpus.

- Segments each recording into 7s windows with 20% overlap.
- Splits data by subject (train/val/test) so no subject leakage.
- Uses subject-level F1 as the primary validation metric and checkpoint criterion.
- Early stopping triggers after patience epochs without subject-F1 improvement
  and only if the best subject accuracy has reached at least 0.8.
- Saves both subject-best and segment-best checkpoints and exports test metrics.
"""

import argparse
import json
import logging
import os
import random
import re
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _safe_model_tag(model_name: str) -> str:
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
    batch_size: int = 32
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
    pretrained_freeze_epochs: int = 5
    pretrained_finetune_epochs: int = 45
    label_smoothing: float = 0.1
    best_segment_model_path: str = "best_segment_model_wavlm-large.pt"
    best_subject_model_path: str = "best_subject_model_wavlm-large.pt"
    patience: int = 5
    log_dir: str = "logs_wavlm-large"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = ExperimentConfig()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging() -> None:
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)


def natural_key(name: str) -> Tuple[int, str]:
    base = os.path.splitext(os.path.basename(name))[0]
    m = re.search(r"\d+", base)
    return (int(m.group()), base.lower()) if m else (float("inf"), base.lower())


def safe_audio_info(path: str) -> Tuple[int, int]:
    try:
        info = torchaudio.info(path)
        return info.num_frames, info.sample_rate
    except Exception:
        with sf.SoundFile(path) as f:
            return len(f), f.samplerate


def safe_audio_load(path: str) -> Tuple[torch.Tensor, int]:
    try:
        wav, sr = torchaudio.load(path)
        return wav, sr
    except Exception:
        data, sr = sf.read(path, always_2d=True)
        return torch.from_numpy(data.T).float(), sr


def normalize_peak_amplitude(x: torch.Tensor) -> torch.Tensor:
    peak = x.abs().max()
    if peak > 1e-8:
        x = x / peak
    return x.clamp(-1.0, 1.0)


def trim_silence(x: torch.Tensor, sr: int, frame_ms: int, hop_ms: int, thr: float) -> torch.Tensor:
    fl = max(1, int(sr * frame_ms / 1000))
    hl = max(1, int(sr * hop_ms / 1000))
    if x.numel() < fl:
        return x
    frames = x.unfold(0, fl, hl)
    energies = frames.pow(2).mean(-1)
    mask = energies > thr
    if not mask.any():
        return x
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    st = int(idx[0]) * hl
    ed = int(idx[-1]) * hl + fl
    return x[st:min(ed, x.size(0))]


def median_filter_1d(x: torch.Tensor, k: int) -> torch.Tensor:
    if k < 3 or k % 2 == 0 or x.numel() < k:
        return x
    pad = k // 2
    padded = F.pad(x.view(1, 1, -1), (pad, pad), mode="reflect").view(-1)
    win = padded.unfold(0, k, 1)
    return win.median(-1).values


def preprocess_waveform(wav: torch.Tensor, cfg: DataConfig) -> torch.Tensor:
    if wav.dim() == 2:
        if wav.size(0) > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
    elif wav.dim() == 0:
        wav = wav.view(1)
    if cfg.normalize_amplitude:
        wav = normalize_peak_amplitude(wav)
    if cfg.apply_silence_trim:
        wav = trim_silence(wav, cfg.sample_rate, cfg.silence_frame_ms, cfg.silence_hop_ms, cfg.silence_energy_threshold)
    if cfg.apply_median_filter:
        wav = median_filter_1d(wav, cfg.median_filter_kernel)
    if wav.numel() == 0:
        wav = torch.zeros(int(cfg.sample_rate * 0.5), dtype=torch.float32)
    return wav.reshape(-1).contiguous()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_metadata(data_cfg: DataConfig) -> Tuple[List[str], List[int], List[str]]:
    excel = os.path.join(data_cfg.data_dir, "subjects_information_audio_lanzhou_2015.xlsx")
    if not os.path.exists(excel):
        raise FileNotFoundError(excel)
    df = pd.read_excel(excel)
    subj_col = next((c for c in df.columns if "subject" in c.lower() or c.lower() == "id"), None)
    label_col = next((c for c in df.columns if c.lower() in {"label", "type"}), None)
    if subj_col is None or label_col is None:
        raise KeyError(f"Columns not found. Available: {list(df.columns)}")
    label_map = {"HC": 0, "MDD": 1}
    paths: List[str] = []
    labels: List[int] = []
    subjects: List[str] = []
    for _, row in df.iterrows():
        raw_id = str(row[subj_col]).strip()
        sid = raw_id.zfill(8) if raw_id.isdigit() else raw_id
        lbl = label_map.get(str(row[label_col]).strip())
        if lbl is None:
            continue
        subj_dir = os.path.join(data_cfg.data_dir, sid)
        if not os.path.isdir(subj_dir):
            logger.warning("Missing subject dir: %s", subj_dir)
            continue
        wavs: List[str] = []
        for root, _, files in os.walk(subj_dir):
            wavs.extend(os.path.join(root, f) for f in files if f.lower().endswith(".wav"))
        for wav_path in sorted(wavs, key=natural_key):
            paths.append(wav_path)
            labels.append(lbl)
            subjects.append(sid)
    logger.info("Loaded %d wavs from %d subjects (HC=%d, MDD=%d)", len(paths), len(set(subjects)), labels.count(0), labels.count(1))
    return paths, labels, subjects


def build_segment_samples(data: Dict[str, List], data_cfg: DataConfig) -> List[Dict]:
    seg_len = data_cfg.sample_rate * data_cfg.segment_duration
    hop = max(1, int(seg_len * (1 - data_cfg.overlap_ratio)))
    samples: List[Dict] = []
    subj_counts = Counter()
    for path, lbl, subj in tqdm(list(zip(data["paths"], data["labels"], data["subjects"])), desc="Indexing audio", total=len(data["paths"])):
        try:
            num_frames, _ = safe_audio_info(path)
        except Exception as exc:
            logger.warning("Skipping %s | %s", path, exc)
            continue
        if num_frames <= 0:
            continue
        max_seg = data_cfg.max_segments_per_subject
        if num_frames <= seg_len:
            samples.append({"path": path, "label": lbl, "subject": subj, "offset": 0})
            subj_counts[subj] += 1
            continue
        added = 0
        for start in range(0, num_frames - seg_len + 1, hop):
            if max_seg is not None and subj_counts[subj] >= max_seg:
                break
            samples.append({"path": path, "label": lbl, "subject": subj, "offset": start})
            subj_counts[subj] += 1
            added += 1
        if added == 0 and (max_seg is None or subj_counts[subj] < max_seg):
            samples.append({"path": path, "label": lbl, "subject": subj, "offset": 0})
            subj_counts[subj] += 1
    return samples


def split_by_subject(file_paths: List[str], labels: List[int], subjects: List[str], train_cfg: TrainConfig) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]:
    subj_to_label: Dict[str, int] = {}
    for subj, lbl in zip(subjects, labels):
        subj_to_label.setdefault(subj, lbl)

    rng = random.Random(train_cfg.random_seed)
    hc = [s for s, l in subj_to_label.items() if l == 0]
    md = [s for s, l in subj_to_label.items() if l == 1]
    rng.shuffle(hc)
    rng.shuffle(md)

    def _split(bucket: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n_train = int(len(bucket) * train_cfg.train_ratio)
        n_val = int(len(bucket) * train_cfg.val_ratio)
        train = bucket[:n_train]
        val = bucket[n_train:n_train + n_val]
        test = bucket[n_train + n_val:]
        return train, val, test

    tr_subj, va_subj, te_subj = set(), set(), set()
    for subset in (_split(hc), _split(md)):
        tr_subj.update(subset[0])
        va_subj.update(subset[1])
        te_subj.update(subset[2])

    def _collect(target: set) -> Dict[str, List]:
        data = {"paths": [], "labels": [], "subjects": []}
        for p, y, s in zip(file_paths, labels, subjects):
            if s in target:
                data["paths"].append(p)
                data["labels"].append(y)
                data["subjects"].append(s)
        return data

    return _collect(tr_subj), _collect(va_subj), _collect(te_subj)


# ---------------------------------------------------------------------------
# Dataset
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
    def __init__(self, samples: List[Dict], data_cfg: DataConfig, split: str) -> None:
        self.data_cfg = data_cfg
        self.split = split
        self.segment_length = data_cfg.sample_rate * data_cfg.segment_duration
        self.apply_augmentation = data_cfg.apply_augmentation and split == "train"
        self.samples = samples
        self.class_counts = Counter(s["label"] for s in samples)
        self.subject_counts = Counter(s["subject"] for s in samples)
        weights = []
        for s in samples:
            class_w = 1.0 / self.class_counts[s["label"]]
            subj_w = 1.0 / self.subject_counts[s["subject"]]
            weights.append(class_w * subj_w)
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.mean() if len(weights) else weights
        self.sample_weights = torch.from_numpy(weights).double()
        self._cache_path: Optional[str] = None
        self._cache_waveform: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_waveform(self, path: str) -> torch.Tensor:
        if path != self._cache_path:
            wav, sr = safe_audio_load(path)
            if sr != self.data_cfg.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.data_cfg.sample_rate)
            wav = preprocess_waveform(wav.squeeze(0), self.data_cfg)
            self._cache_path = path
            self._cache_waveform = wav.contiguous()
        return self._cache_waveform.clone()

    def _crop_segment(self, wav: torch.Tensor, offset: int) -> torch.Tensor:
        target = self.segment_length
        if wav.size(0) <= target:
            return F.pad(wav, (0, target - wav.size(0)))
        offset = min(max(offset, 0), wav.size(0) - target)
        return wav[offset:offset + target]

    def _time_stretch(self, segment: torch.Tensor) -> torch.Tensor:
        min_rate, max_rate = self.data_cfg.time_stretch_range
        if min_rate <= 0 or max_rate <= 0:
            return segment
        rate = random.uniform(min_rate, max_rate)
        if abs(rate - 1.0) < 1e-2:
            return segment
        new_sr = max(1, int(self.data_cfg.sample_rate * rate))
        stretched = torchaudio.functional.resample(segment.unsqueeze(0), self.data_cfg.sample_rate, new_sr).squeeze(0)
        if stretched.size(0) > segment.size(0):
            start = random.randint(0, stretched.size(0) - segment.size(0))
            stretched = stretched[start:start + segment.size(0)]
        elif stretched.size(0) < segment.size(0):
            stretched = F.pad(stretched, (0, segment.size(0) - stretched.size(0)))
        return stretched

    def _time_shift(self, segment: torch.Tensor) -> torch.Tensor:
        max_ratio = self.data_cfg.time_shift_max_ratio
        if max_ratio <= 0:
            return segment
        shift = random.randint(-int(segment.size(0) * max_ratio), int(segment.size(0) * max_ratio))
        if shift == 0:
            return segment
        return torch.roll(segment, shifts=shift)

    def _time_dropout(self, segment: torch.Tensor) -> torch.Tensor:
        max_ratio = self.data_cfg.time_dropout_max_ratio
        if max_ratio <= 0:
            return segment
        span = random.randint(1, max(1, int(segment.size(0) * max_ratio)))
        start = random.randint(0, max(0, segment.size(0) - span))
        dropped = segment.clone()
        dropped[start:start + span] = 0.0
        return dropped

    def _augment(self, segment: torch.Tensor) -> torch.Tensor:
        if random.random() > self.data_cfg.augmentation_prob:
            return segment
        if random.random() < self.data_cfg.time_stretch_prob:
            segment = self._time_stretch(segment)
        if random.random() < self.data_cfg.time_shift_prob:
            segment = self._time_shift(segment)
        if random.random() < self.data_cfg.time_dropout_prob:
            segment = self._time_dropout(segment)
        if random.random() < 0.6:
            std = random.uniform(*self.data_cfg.noise_std_range)
            segment = segment + torch.randn_like(segment) * std
        if random.random() < 0.5:
            gain = random.uniform(*self.data_cfg.gain_range)
            segment = segment * gain
        return segment.clamp(-1.0, 1.0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, int]:
        sample = self.samples[idx]
        wav = self._load_waveform(sample["path"])
        segment = self._crop_segment(wav, sample["offset"])
        if self.apply_augmentation:
            segment = self._augment(segment)
        return segment.float(), sample["label"], sample["subject"], segment.numel()


def collate_fn(batch: List[Tuple[torch.Tensor, int, str, int]], data_cfg: DataConfig, model_cfg: ModelConfig):
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
# Model
# ---------------------------------------------------------------------------

class WavLMClassifier(nn.Module):
    def __init__(self, model_cfg: ModelConfig, num_classes: int) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = WavLMModel.from_pretrained(
            model_cfg.model_name,
            cache_dir=model_cfg.hf_cache_dir,
            local_files_only=model_cfg.local_files_only,
        )
        hidden = self.backbone.config.hidden_size
        hidden_dim = model_cfg.classifier_hidden_dim or hidden
        self.frame_dropout = nn.Dropout(model_cfg.frame_dropout)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        if not trainable:
            return
        if hasattr(self.backbone, "feature_extractor"):
            for p in self.backbone.feature_extractor.parameters():
                p.requires_grad = False
        layers = getattr(self.backbone.encoder, "layers", None)
        if layers is None:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            n = len(layers)
            k = self.model_cfg.unfreeze_last_n_layers or n
            for layer in layers[-k:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if hasattr(self.backbone, "layer_norm"):
            for p in self.backbone.layer_norm.parameters():
                p.requires_grad = True
        self.backbone.train()

    def head_parameters(self) -> List[nn.Parameter]:
        params = list(self.norm.parameters()) + list(self.classifier.parameters())
        return [p for p in params if p.requires_grad]

    def backbone_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = batch.get("attention_mask")
        outputs = self.backbone(batch["input_values"], attention_mask=attn, output_hidden_states=False)
        hidden = self.frame_dropout(outputs.last_hidden_state)
        if attn is None:
            mask = torch.ones(hidden.size()[:2], device=hidden.device, dtype=torch.long)
        else:
            mask = attn.to(hidden.device)
        input_lengths = mask.sum(dim=1)
        if hasattr(self.backbone, "_get_feat_extract_output_lengths"):
            feat_lengths = self.backbone._get_feat_extract_output_lengths(input_lengths).to(hidden.device)
        else:
            stride = int(np.prod(getattr(self.backbone.config, "conv_stride", [1])))
            feat_lengths = torch.div(input_lengths + stride - 1, stride, rounding_mode="floor").to(hidden.device)
        max_len = hidden.size(1)
        frame_index = torch.arange(max_len, device=hidden.device).unsqueeze(0)
        frame_mask = frame_index < feat_lengths.unsqueeze(1)
        mask_f = frame_mask.unsqueeze(-1).type_as(hidden)
        pooled = (hidden * mask_f).sum(dim=1) / frame_mask.sum(dim=1, keepdim=True).clamp(min=1).type_as(hidden)
        pooled = self.norm(pooled)
        logits = self.classifier(self.dropout(pooled))
        return logits, pooled


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def evaluate(model: WavLMClassifier, dataloader: DataLoader, device: torch.device, use_amp: bool, split_name: str) -> Dict[str, float]:
    model.eval()
    all_logits: List[List[float]] = []
    all_labels: List[int] = []
    all_subjects: List[str] = []
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch, labels, subjects in tqdm(dataloader, desc=f"Evaluating[{split_name}]", leave=False):
            labels = labels.to(device)
            batch = move_batch_to_device(batch, device)
            with autocast(enabled=use_amp):
                logits, _ = model(batch)
                probs = torch.softmax(logits, dim=1)
            loss = F.cross_entropy(logits.float(), labels, reduction="sum", label_smoothing=CONFIG.train.label_smoothing)
            total_loss += loss.item()
            total_samples += labels.size(0)
            all_logits.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_subjects.extend(subjects)

    metrics: Dict[str, float] = {
        "segment_acc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auc": float("nan"),
        "loss": total_loss / max(total_samples, 1),
        "confusion_matrix": None,
        "subject_acc": 0.0,
        "subject_precision": 0.0,
        "subject_recall": 0.0,
        "subject_f1": 0.0,
        "subject_auc": float("nan"),
    }
    if not all_labels:
        return metrics

    preds = np.argmax(np.array(all_logits), axis=1)
    probs_pos = np.array(all_logits)[:, 1]
    metrics["segment_acc"] = accuracy_score(all_labels, preds)
    metrics["precision"] = precision_score(all_labels, preds, zero_division=0)
    metrics["recall"] = recall_score(all_labels, preds, zero_division=0)
    metrics["f1"] = f1_score(all_labels, preds, zero_division=0)
    try:
        metrics["auc"] = roc_auc_score(all_labels, probs_pos) if len(set(all_labels)) > 1 else float("nan")
    except ValueError:
        metrics["auc"] = float("nan")
    metrics["confusion_matrix"] = confusion_matrix(all_labels, preds, labels=[0, 1])

    subject_logits: Dict[str, List[List[float]]] = defaultdict(list)
    subject_true: Dict[str, int] = {}
    for logit, label, subj in zip(all_logits, all_labels, all_subjects):
        subject_logits[subj].append(logit)
        subject_true[subj] = label
    subject_preds: List[int] = []
    subject_labels: List[int] = []
    subject_probs: List[float] = []
    for subj, logits_list in subject_logits.items():
        logits_tensor = torch.tensor(logits_list, dtype=torch.float32)
        mean_logits = logits_tensor.mean(dim=0)
        mean_probs = torch.softmax(mean_logits, dim=0)
        subject_preds.append(int(torch.argmax(mean_probs).item()))
        subject_probs.append(float(mean_probs[1].item()))
        subject_labels.append(subject_true[subj])

    if subject_labels:
        metrics["subject_acc"] = accuracy_score(subject_labels, subject_preds)
        metrics["subject_precision"] = precision_score(subject_labels, subject_preds, zero_division=0)
        metrics["subject_recall"] = recall_score(subject_labels, subject_preds, zero_division=0)
        metrics["subject_f1"] = f1_score(subject_labels, subject_preds, zero_division=0)
        try:
            metrics["subject_auc"] = roc_auc_score(subject_labels, subject_probs) if len(set(subject_labels)) > 1 else float("nan")
        except ValueError:
            metrics["subject_auc"] = float("nan")

    logger.info(
        "[%s] Segment acc=%.4f | P=%.4f | R=%.4f | F1=%.4f | AUC=%.4f | loss=%.4f",
        split_name,
        metrics["segment_acc"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auc"],
        metrics["loss"],
    )
    logger.info(
        "[%s] Subject acc=%.4f | P=%.4f | R=%.4f | F1=%.4f | AUC=%.4f",
        split_name,
        metrics["subject_acc"],
        metrics["subject_precision"],
        metrics["subject_recall"],
        metrics["subject_f1"],
        metrics["subject_auc"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_dataloader(dataset: AudioDataset, data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig, sampler: Optional[WeightedRandomSampler], shuffle: bool) -> DataLoader:
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


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def run_training() -> None:
    setup_logging()
    if CONFIG.model.force_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.makedirs(CONFIG.model.hf_cache_dir, exist_ok=True)
    os.makedirs(CONFIG.train.log_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", CONFIG.model.hf_cache_dir)
    seed_everything(CONFIG.train.random_seed)

    file_paths, labels, subjects = load_metadata(CONFIG.data)
    train_split, val_split, test_split = split_by_subject(file_paths, labels, subjects, CONFIG.train)

    # build segments within each split
    train_samples = build_segment_samples(train_split, CONFIG.data)
    val_samples = build_segment_samples(val_split, CONFIG.data)
    test_samples = build_segment_samples(test_split, CONFIG.data)

    rng = random.Random(CONFIG.train.random_seed)
    rng.shuffle(train_samples)

    subj_label_map: Dict[str, int] = {}
    for split in (train_split, val_split, test_split):
        for subj, lbl in zip(split["subjects"], split["labels"]):
            subj_label_map.setdefault(subj, lbl)
    subj_counts = Counter(subj_label_map.values())
    logger.info("Subjects | HC=%d | MDD=%d", subj_counts.get(0, 0), subj_counts.get(1, 0))
    seg_counts = Counter(s["label"] for s in train_samples + val_samples + test_samples)
    logger.info("Segments | HC=%d | MDD=%d | total=%d", seg_counts[0], seg_counts[1], len(train_samples) + len(val_samples) + len(test_samples))

    train_dataset = AudioDataset(train_samples, CONFIG.data, split="train")
    val_dataset = AudioDataset(val_samples, CONFIG.data, split="val")
    test_dataset = AudioDataset(test_samples, CONFIG.data, split="test")

    sampler = None
    if train_dataset.sample_weights.numel() > 0:
        sampler = WeightedRandomSampler(train_dataset.sample_weights, num_samples=len(train_dataset), replacement=False)

    train_loader = build_dataloader(train_dataset, CONFIG.data, CONFIG.model, CONFIG.train, sampler, shuffle=True)
    val_loader = build_dataloader(val_dataset, CONFIG.data, CONFIG.model, CONFIG.train, sampler=None, shuffle=False)
    test_loader = build_dataloader(test_dataset, CONFIG.data, CONFIG.model, CONFIG.train, sampler=None, shuffle=False)

    model = WavLMClassifier(CONFIG.model, num_classes=2).to(CONFIG.device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.head_parameters(), "lr": CONFIG.train.head_learning_rate},
        ],
        weight_decay=CONFIG.train.weight_decay,
    )
    total_epochs = CONFIG.train.pretrained_freeze_epochs + CONFIG.train.pretrained_finetune_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=CONFIG.train.scheduler_eta_min)
    scaler = GradScaler(enabled=CONFIG.train.use_amp and CONFIG.device.type == "cuda")

    best_subject_f1 = float("-inf")
    best_subject_acc = 0.0
    best_segment_f1 = float("-inf")
    patience = 0
    backbone_added = False

    for epoch in range(1, total_epochs + 1):
        logger.info("Epoch %d/%d", epoch, total_epochs)
        if not backbone_added and epoch > CONFIG.train.pretrained_freeze_epochs:
            model.set_backbone_trainable(True)
            bb_params = model.backbone_parameters()
            if bb_params:
                optimizer.add_param_group({"params": bb_params, "lr": CONFIG.train.backbone_learning_rate})
                scheduler.base_lrs.append(CONFIG.train.backbone_learning_rate)
                backbone_added = True
                logger.info("Unfroze last %d transformer layers", CONFIG.model.unfreeze_last_n_layers)

        model.train()
        total_loss = 0.0
        correct = 0
        total_examples = 0
        train_labels: List[int] = []
        train_preds: List[int] = []
        amp_enabled = scaler.is_enabled()
        progress = tqdm(train_loader, desc="Training", leave=False)
        for batch, labels, _ in progress:
            labels = labels.to(CONFIG.device)
            batch = move_batch_to_device(batch, CONFIG.device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                logits, _ = model(batch)
                loss = F.cross_entropy(logits, labels, label_smoothing=CONFIG.train.label_smoothing)
            scaler.scale(loss).backward()
            if CONFIG.train.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), CONFIG.train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_examples += labels.size(0)
            total_loss += loss.item()
            train_preds.extend(preds.detach().cpu().tolist())
            train_labels.extend(labels.detach().cpu().tolist())
            running_f1 = f1_score(train_labels, train_preds, zero_division=0) if train_labels else 0.0
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                f1=f"{running_f1:.3f}",
            )

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc = correct / total_examples if total_examples else 0.0
        train_f1 = f1_score(train_labels, train_preds, zero_division=0) if train_labels else 0.0
        logger.info("Train | loss=%.4f | acc=%.4f | f1=%.4f", train_loss, train_acc, train_f1)

        val_metrics = evaluate(model, val_loader, CONFIG.device, use_amp=False, split_name=f"val-epoch{epoch}")
        subject_f1 = val_metrics.get("subject_f1", 0.0)
        subject_acc = val_metrics.get("subject_acc", 0.0)
        segment_f1 = val_metrics.get("f1", 0.0)

        improved = False
        if subject_f1 > best_subject_f1:
            best_subject_f1 = subject_f1
            best_subject_acc = subject_acc
            torch.save(model.state_dict(), CONFIG.train.best_subject_model_path)
            logger.info("Saved best subject model (epoch %d, subject_f1=%.4f) -> %s", epoch, subject_f1, CONFIG.train.best_subject_model_path)
            improved = True
        if segment_f1 > best_segment_f1:
            best_segment_f1 = segment_f1
            torch.save(model.state_dict(), CONFIG.train.best_segment_model_path)
            logger.info("Saved best segment model (epoch %d, f1=%.4f) -> %s", epoch, segment_f1, CONFIG.train.best_segment_model_path)
            improved = True

        if improved:
            patience = 0
        else:
            patience += 1
            logger.info("No metric improvement for %d epoch(s)", patience)
            if patience >= CONFIG.train.patience and best_subject_acc >= 0.8:
                logger.info("Early stopping triggered (best subject acc=%.4f)", best_subject_acc)
                break

        scheduler.step()

    logger.info("Evaluating on test set")
    test_metrics: Dict[str, Dict[str, float]] = {}
    if os.path.exists(CONFIG.train.best_segment_model_path):
        model.load_state_dict(torch.load(CONFIG.train.best_segment_model_path, map_location=CONFIG.device))
        seg_best = evaluate(model, test_loader, CONFIG.device, use_amp=False, split_name="test-segment-best")
        test_metrics["segment_best"] = seg_best
    if os.path.exists(CONFIG.train.best_subject_model_path):
        model.load_state_dict(torch.load(CONFIG.train.best_subject_model_path, map_location=CONFIG.device))
        subj_best = evaluate(model, test_loader, CONFIG.device, use_amp=False, split_name="test-subject-best")
        test_metrics["subject_best"] = subj_best

    metrics_path = os.path.join(CONFIG.train.log_dir, f"test_metrics_subject_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(test_metrics), f, ensure_ascii=False, indent=2)
    logger.info("Saved test metrics to %s", metrics_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WavLM-Large with subject split")
    parser.add_argument("--data_dir", default=CONFIG.data.data_dir)
    parser.add_argument("--batch_size", type=int, default=CONFIG.train.batch_size)
    parser.add_argument("--epochs", type=int, default=CONFIG.train.pretrained_freeze_epochs + CONFIG.train.pretrained_finetune_epochs)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    CONFIG.data.data_dir = args.data_dir
    CONFIG.train.batch_size = args.batch_size
    CONFIG.train.pretrained_finetune_epochs = max(0, args.epochs - CONFIG.train.pretrained_freeze_epochs)
    run_training()


if __name__ == "__main__":
    main()
