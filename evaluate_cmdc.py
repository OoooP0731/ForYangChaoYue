#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a pretrained wav2vec2/WavLM classifier on the CMDC corpus.

This script loads the segmentation-based classifier that was trained on the
MODMA dataset (see the accompanying training script) and runs it on the CMDC
dataset using the checkpoint ``best_model_wavlm_lagre.pt`` by default.  The
evaluation operates at the *segment* level – every 7‑second clip is treated as
an individual sample – and reports accuracy, precision, recall, F1 score, and
ROC AUC.

Usage
-----
```
python evaluate_cmdc.py \
    --cmdc_dir /path/to/CMDC \
    --checkpoint best_model_wavlm_lagre.pt \
    --model_name microsoft/wavlm-large
```

The script assumes the CMDC directory follows the published structure, where
top-level folders correspond to diagnostic groups (e.g. ``MDD`` or ``HC``) and
contain subject folders with ``Q*.wav`` recordings.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, WavLMModel


@dataclass
class DataConfig:
    """Configuration controlling data preparation."""

    sample_rate: int = 16_000
    segment_duration: int = 7  # seconds
    overlap_ratio: float = 0.2
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
    """Configuration describing the classifier architecture."""

    model_name: str = "microsoft/wavlm-large"
    hf_cache_dir: str = "./hf_cache"
    local_files_only: bool = False
    dropout: float = 0.3
    frame_dropout: float = 0.1
    classifier_hidden_dim: int = 256


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# CMDC metadata and segment construction
# ---------------------------------------------------------------------------


def discover_cmdc_segments(
    cmdc_dir: str,
    data_cfg: DataConfig,
) -> List[Dict[str, object]]:
    """Scan the CMDC directory and assemble segment metadata.

    Each subject folder is expected to live inside a group directory whose
    name contains ``MDD`` or ``HC``.  All ``.wav`` files beneath the subject
    directory are included.  The function returns a list of dictionaries with
    keys ``path`` (absolute path), ``label`` (0 or 1), ``subject`` (identifier),
    and ``offset`` (starting frame for the 7-second segment).  Longer files are
    split using the configured hop length (based on ``overlap_ratio``).
    """

    if not os.path.isdir(cmdc_dir):
        raise FileNotFoundError(f"CMDC directory not found: {cmdc_dir}")

    segment_length = data_cfg.sample_rate * data_cfg.segment_duration
    hop_length = max(1, int(segment_length * (1.0 - data_cfg.overlap_ratio)))

    samples: List[Dict[str, object]] = []
    subject_counts: Dict[str, int] = {}

    for group_name in sorted(os.listdir(cmdc_dir)):
        group_path = os.path.join(cmdc_dir, group_name)
        if not os.path.isdir(group_path):
            continue
        name_upper = group_name.upper()
        if "MDD" in name_upper:
            label = 1
        elif "HC" in name_upper:
            label = 0
        else:
            # Skip auxiliary folders that do not correspond to a diagnostic group.
            continue

        for subject_name in sorted(os.listdir(group_path)):
            subject_path = os.path.join(group_path, subject_name)
            if not os.path.isdir(subject_path):
                continue
            subject_id = f"{group_name}/{subject_name}"
            subject_counts.setdefault(subject_id, 0)

            wav_files: List[str] = []
            for root, _, files in os.walk(subject_path):
                for file in files:
                    if file.lower().endswith(".wav"):
                        wav_files.append(os.path.join(root, file))
            wav_files.sort()

            for wav_path in wav_files:
                num_frames, sample_rate = safe_audio_info(wav_path)
                if num_frames <= 0:
                    continue

                if sample_rate != data_cfg.sample_rate and sample_rate > 0:
                    resampled_frames = int(round(num_frames * data_cfg.sample_rate / sample_rate))
                else:
                    resampled_frames = num_frames

                if resampled_frames <= segment_length:
                    samples.append(
                        {
                            "path": wav_path,
                            "label": label,
                            "subject": subject_id,
                            "offset": 0,
                        }
                    )
                    subject_counts[subject_id] += 1
                    continue

                start_positions = list(range(0, max(resampled_frames - segment_length + 1, 1), hop_length))
                if not start_positions:
                    start_positions = [0]

                for offset in start_positions:
                    max_segments = data_cfg.max_segments_per_subject
                    if max_segments is not None and subject_counts[subject_id] >= max_segments:
                        break
                    samples.append(
                        {
                            "path": wav_path,
                            "label": label,
                            "subject": subject_id,
                            "offset": offset,
                        }
                    )
                    subject_counts[subject_id] += 1

    if not samples:
        raise RuntimeError(
            "No audio segments were discovered. Please verify the CMDC directory "
            "structure and that .wav files are present."
        )

    return samples


# ---------------------------------------------------------------------------
# Audio preprocessing utilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Dataset definition and collate function
# ---------------------------------------------------------------------------


class CmdcDataset(Dataset):
    def __init__(self, samples: List[Dict[str, object]], data_cfg: DataConfig):
        self.samples = samples
        self.data_cfg = data_cfg
        self.segment_length = data_cfg.sample_rate * data_cfg.segment_duration
        self._cache_path: Optional[str] = None
        self._cache_waveform: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_waveform(self, path: str) -> torch.Tensor:
        if path != self._cache_path:
            waveform, sample_rate = safe_audio_load(path)
            if sample_rate != self.data_cfg.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, self.data_cfg.sample_rate
                )
            waveform = preprocess_waveform(waveform.squeeze(0), self.data_cfg)
            self._cache_path = path
            self._cache_waveform = waveform.contiguous()
        return self._cache_waveform.clone()

    def _crop_segment(self, waveform: torch.Tensor, offset: int) -> torch.Tensor:
        target = self.segment_length
        if waveform.size(0) <= target:
            return F.pad(waveform, (0, target - waveform.size(0)))
        offset = int(min(max(offset, 0), waveform.size(0) - target))
        return waveform[offset:offset + target]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        waveform = self._load_waveform(sample["path"])  # type: ignore[arg-type]
        segment = self._crop_segment(waveform, sample["offset"])  # type: ignore[arg-type]
        return segment.float(), int(sample["label"])


def build_feature_extractor(model_cfg: ModelConfig) -> Wav2Vec2FeatureExtractor:
    return Wav2Vec2FeatureExtractor.from_pretrained(
        model_cfg.model_name,
        cache_dir=model_cfg.hf_cache_dir,
        local_files_only=model_cfg.local_files_only,
    )


def collate_fn(
    batch: Iterable[Tuple[torch.Tensor, int]],
    extractor: Wav2Vec2FeatureExtractor,
    data_cfg: DataConfig,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    segments, labels = zip(*batch)
    inputs = extractor(
        [seg.numpy() for seg in segments],
        sampling_rate=data_cfg.sample_rate,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    attention_mask = inputs.attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(inputs.input_values, dtype=torch.long)
    else:
        attention_mask = attention_mask.long()
    return (
        {
            "input_values": inputs.input_values,
            "attention_mask": attention_mask,
        },
        torch.tensor(labels, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class WavClassifier(nn.Module):
    def __init__(self, model_cfg: ModelConfig, num_classes: int = 2):
        super().__init__()
        if "wavlm" in model_cfg.model_name.lower():
            self.backbone = WavLMModel.from_pretrained(
                model_cfg.model_name,
                cache_dir=model_cfg.hf_cache_dir,
                local_files_only=model_cfg.local_files_only,
            )
        else:
            self.backbone = Wav2Vec2Model.from_pretrained(
                model_cfg.model_name,
                cache_dir=model_cfg.hf_cache_dir,
                local_files_only=model_cfg.local_files_only,
            )

        hidden_size = int(self.backbone.config.hidden_size)
        hidden_dim = model_cfg.classifier_hidden_dim or hidden_size

        self.frame_dropout = nn.Dropout(model_cfg.frame_dropout)
        self.pre_classifier_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.backbone(
            batch["input_values"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=False,
        )
        hidden_states = self.frame_dropout(outputs.last_hidden_state)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            mask = torch.ones(
                hidden_states.size()[:2],
                device=hidden_states.device,
                dtype=torch.long,
            )
        else:
            mask = attention_mask.to(hidden_states.device)

        input_lengths = mask.sum(dim=1)
        if hasattr(self.backbone, "_get_feat_extract_output_lengths"):
            feat_lengths = self.backbone._get_feat_extract_output_lengths(input_lengths)
            feat_lengths = feat_lengths.to(hidden_states.device)
        else:
            stride = getattr(self.backbone.config, "conv_stride", None)
            if stride is None and hasattr(self.backbone, "feature_extractor"):
                stride = [layer.stride[0] for layer in self.backbone.feature_extractor.conv_layers]
            total_stride = int(np.prod(stride)) if stride else 1
            feat_lengths = torch.div(
                input_lengths + total_stride - 1,
                total_stride,
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
        return self.classifier(self.dropout(pooled))


# ---------------------------------------------------------------------------
# Evaluation routine
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            labels = labels.to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    metrics: Dict[str, float] = {
        "ACC": 0.0,
        "Pre": 0.0,
        "Rec": 0.0,
        "F1": 0.0,
        "AUC": float("nan"),
    }

    if not all_labels:
        return metrics

    metrics["ACC"] = accuracy_score(all_labels, all_preds)
    metrics["Pre"] = precision_score(all_labels, all_preds, zero_division=0)
    metrics["Rec"] = recall_score(all_labels, all_preds, zero_division=0)
    metrics["F1"] = f1_score(all_labels, all_preds, zero_division=0)

    try:
        if len(set(all_labels)) > 1:
            metrics["AUC"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["AUC"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:  # pragma: no cover - interactive environments may not define __file__
    SCRIPT_DIR = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pretrained model on CMDC")
    parser.add_argument(
        "--cmdc_dir",
        type=str,
        default=None,
        help="Path to the CMDC dataset root (defaults to ./CMDC next to this script)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(SCRIPT_DIR / "best_model_wavlm_lagre.pt"),
        help="Checkpoint file produced by MODMA training",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/wavlm-large",
        help="Hugging Face model identifier used during training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Evaluation device (e.g. 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=24,
        help="Random seed for deterministic segment ordering",
    )
    parser.add_argument(
        "--no_cache_download",
        action="store_true",
        help="Allow huggingface to download weights if not cached locally",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    data_cfg = DataConfig()
    model_cfg = ModelConfig(
        model_name=args.model_name,
        local_files_only=not args.no_cache_download,
    )

    os.makedirs(model_cfg.hf_cache_dir, exist_ok=True)
    if model_cfg.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", model_cfg.hf_cache_dir)

    default_cmdc = SCRIPT_DIR / "CMDC"
    cmdc_dir = Path(args.cmdc_dir).expanduser().resolve() if args.cmdc_dir else default_cmdc.resolve()
    if not os.path.isdir(cmdc_dir):
        raise FileNotFoundError(
            "CMDC directory not found. Pass --cmdc_dir or place the dataset in ./CMDC."
        )

    print("Discovering CMDC audio segments...")
    samples = discover_cmdc_segments(str(cmdc_dir), data_cfg)
    random.Random(args.seed).shuffle(samples)
    subject_set = {s["subject"] for s in samples}
    label_counts = Counter(sample["label"] for sample in samples)
    print(
        f"Found {len(samples)} segments across {len(subject_set)} subjects "
        f"(HC segments: {label_counts.get(0, 0)}, MDD segments: {label_counts.get(1, 0)})."
    )

    dataset = CmdcDataset(samples, data_cfg)
    extractor = build_feature_extractor(model_cfg)

    def _collate(batch: Iterable[Tuple[torch.Tensor, int]]):
        return collate_fn(batch, extractor, data_cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = WavClassifier(model_cfg, num_classes=2).to(device)

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (SCRIPT_DIR / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected}")

    metrics = evaluate(model, dataloader, device)
    print("\nCMDC evaluation results (segment-level):")
    for key in ["ACC", "Pre", "Rec", "F1", "AUC"]:
        value = metrics[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}" if not np.isnan(value) else f"  {key}: nan")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

