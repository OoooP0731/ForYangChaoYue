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
import torch.optim as optim
import torchaudio
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _safe_model_tag(model_name: str) -> str:
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
    feature_type: str = "pretrained"  # "pretrained" or "fbank"
    fbank_num_mel: int = 40
    fbank_frame_length_ms: int = 25
    fbank_frame_shift_ms: int = 10


@dataclass
class ModelConfig:
    model_name: str = "microsoft/wavlm-base-plus"
    hidden_dim: int = 768
    unfreeze_last_n_layers: int = 0
    hf_cache_dir: str = "./hf_cache"
    local_files_only: bool = True
    force_offline: bool = True
    use_layer_weighting: bool = True
    ecapa_channels: int = 512
    embedding_dim: int = 192
    stats_dropout: float = 0.3

    @property
    def tag(self) -> str:
        return _safe_model_tag(self.model_name)


@dataclass
class TrainConfig:
    batch_size: int = 128
    num_workers: int = 4
    persistent_workers: bool = True
    head_learning_rate: float = 5e-5
    backbone_learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    scheduler_eta_min: float = 1e-6
    max_grad_norm: float = 1.0
    use_amp: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 24
    resume_from_best: bool = False
    best_model_path: str = "best_model_wavlm-base-plus.pt"
    log_dir: str = "logs_wavlm-base-plus"
    curve_path: str = "training_curves_by_wavlm-base-plus.png"
    val_selection_metric: str = "segment"
    log_eval_details: bool = True
    plot_training_curves: bool = True
    max_nan_warnings: int = 3
    nan_lr_scale: float = 0.5
    disable_amp_on_nan: bool = True
    fbank_epochs: int = 165
    pretrained_freeze_epochs: int = 20
    pretrained_finetune_epochs: int = 5
    aam_margin: float = 0.2
    aam_scale: float = 30.0
    inter_topk: int = 5
    inter_margin: float = 0.1
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
CONFIG.train.best_model_path = f"best_model_{CONFIG.model.tag}.pt"
CONFIG.train.log_dir = f"logs_{CONFIG.model.tag}"
CONFIG.train.curve_path = f"training_curves_by_{CONFIG.model.tag}.png"


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
            wavs.extend(
                os.path.join(root, f)
                for f in files
                if f.lower().endswith(".wav")
            )
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
        val = bucket[n_train : n_train + n_val]
        test = bucket[n_train + n_val :]
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
        len(train_subjects),
        len(train_data["paths"]),
        len(val_subjects),
        len(val_data["paths"]),
        len(test_subjects),
        len(test_data["paths"]),
    )
    return train_data, val_data, test_data


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
        val = bucket[n_train : n_train + n_val]
        test = bucket[n_train + n_val :]
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
        len(train_subjects),
        len(train_data["paths"]),
        len(val_subjects),
        len(val_data["paths"]),
        len(test_subjects),
        len(test_data["paths"]),
    )
    return train_data, val_data, test_data



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
    pad = kernel // 2
    padded = F.pad(waveform.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)
    windows = padded.unfold(0, kernel, 1)
    return windows.median(dim=-1).values


def preprocess_waveform(waveform: torch.Tensor, data_cfg: DataConfig) -> torch.Tensor:
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
    return waveform.contiguous()


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


_feature_extractor: Optional[Wav2Vec2FeatureExtractor] = None


def get_feature_extractor(model_cfg: ModelConfig) -> Wav2Vec2FeatureExtractor:
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_cfg.model_name,
            local_files_only=model_cfg.local_files_only,
            cache_dir=model_cfg.hf_cache_dir,
        )
    return _feature_extractor


class AudioDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, List],
        data_cfg: DataConfig,
    ) -> None:
        self.data_cfg = data_cfg
        self.segment_length = data_cfg.sample_rate * data_cfg.segment_duration
        self.hop_length = max(1, int(self.segment_length * (1 - data_cfg.overlap_ratio)))
        self.samples: List[Dict] = []
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
        self.sample_weights = self._build_weights()
        self._cache_path: Optional[str] = None
        self._cache_waveform: Optional[torch.Tensor] = None

    def _build_weights(self) -> torch.DoubleTensor:
        if not self.samples:
            return torch.DoubleTensor()
        weights = []
        for sample in self.samples:
            class_w = 1.0 / self.class_counts[sample["label"]]
            subject_w = 1.0 / self.subject_counts[sample["subject"]]
            weights.append(class_w * subject_w)
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.mean()
        return torch.from_numpy(weights).double()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_waveform(self, path: str) -> torch.Tensor:
        if path != self._cache_path:
            waveform, sample_rate = safe_audio_load(path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != self.data_cfg.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform,
                    sample_rate,
                    self.data_cfg.sample_rate,
                )
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
        segment = waveform[offset : offset + target]
        return segment, target

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, int]:
        sample = self.samples[index]
        waveform = self._load_waveform(sample["path"])
        segment, length = self._crop_segment(waveform, sample["offset"])
        return segment.float(), sample["label"], sample["subject"], length



def _compute_fbank(segment: torch.Tensor, data_cfg: DataConfig) -> torch.Tensor:
    waveform = segment.unsqueeze(0)
    features = torchaudio.compliance.kaldi.fbank(
        waveform,
        sample_frequency=data_cfg.sample_rate,
        num_mel_bins=data_cfg.fbank_num_mel,
        frame_length=data_cfg.fbank_frame_length_ms,
        frame_shift=data_cfg.fbank_frame_shift_ms,
        use_energy=False,
        dither=0.0,
    )
    return features


def collate_fn(
    batch: List[Tuple[torch.Tensor, int, str, int]],
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
):
    segments, labels, subjects, lengths = zip(*batch)
    segments = [seg.clone() for seg in segments]
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    if data_cfg.feature_type.lower() == "fbank":
        features: List[torch.Tensor] = []
        frame_lengths: List[int] = []
        for segment, length in zip(segments, lengths_tensor.tolist()):
            trimmed = segment[:length]
            fb = _compute_fbank(trimmed, data_cfg)
            features.append(fb)
            frame_lengths.append(fb.size(0))
        padded = nn.utils.rnn.pad_sequence(features, batch_first=True)
        length_tensor = torch.tensor(frame_lengths, dtype=torch.long)
        return (
            {"features": padded, "feature_lengths": length_tensor},
            label_tensor,
            list(subjects),
        )
    extractor = get_feature_extractor(model_cfg)
    segments_np = [seg.numpy() for seg in segments]
    processed = extractor(
        segments_np,
        sampling_rate=data_cfg.sample_rate,
        padding=True,
        return_tensors="pt",
    )
    return (
        {
            "input_values": processed.input_values,
            "attention_mask": processed.attention_mask.long(),
            "sample_lengths": lengths_tensor,
        },
        label_tensor,
        list(subjects),
    )


def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    range_tensor = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return range_tensor < lengths.unsqueeze(1)


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(F.relu(self.conv(x)))


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels: int, kernel_size: int, scale: int = 8, dilation: int = 1) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError("scale must be >= 1")
        if scale > 1 and channels % scale != 0:
            raise ValueError("channels must be divisible by scale when scale > 1")
        self.scale = scale
        self.width = channels // scale if scale > 1 else channels
        self.nums = scale
        padding = dilation * (kernel_size - 1) // 2
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
                for _ in range(self.nums - 1)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums - 1)])
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 1:
            if not self.convs:
                return x
            out = self.convs[0](x)
            out = self.relu(out)
            return self.bns[0](out)
        splits = torch.split(x, self.width, dim=1)
        outputs = []
        for i in range(self.nums):
            if i == 0:
                outputs.append(splits[i])
            else:
                temp = splits[i] + outputs[i - 1]
                temp = self.convs[i - 1](temp)
                temp = self.relu(temp)
                temp = self.bns[i - 1](temp)
                outputs.append(temp)
        return torch.cat(outputs, dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.conv2 = nn.Conv1d(bottleneck, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=2, keepdim=True)
        excitation = torch.sigmoid(self.conv2(F.relu(self.conv1(pooled))))
        return x * excitation


class SERes2Block(nn.Module):
    def __init__(self, channels: int, kernel_size: int, scale: int, dilation: int) -> None:
        super().__init__()
        self.res2 = Res2Conv1dReluBn(channels, kernel_size, scale=scale, dilation=dilation)
        self.conv1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.res2(x)
        out = self.conv1x1(out)
        out = self.bn(out)
        out = self.se(out)
        return out + residual


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim: int, attention_channels: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, input_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        max_len = x.size(2)
        mask = lengths_to_mask(lengths, max_len).unsqueeze(1)
        attn_logits = self.attention(x).masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(attn_logits, dim=2) * mask
        attn = attn / (attn.sum(dim=2, keepdim=True) + 1e-9)
        mean = torch.sum(x * attn, dim=2)
        var = torch.sum(((x - mean.unsqueeze(-1)) ** 2) * attn, dim=2)
        std = torch.sqrt(torch.clamp(var, min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN_Small(nn.Module):
    def __init__(self, input_dim: int, channels: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.layer1 = Conv1dReluBn(input_dim, channels, kernel_size=5)
        self.layer2 = SERes2Block(channels, kernel_size=3, scale=8, dilation=2)
        self.layer3 = SERes2Block(channels, kernel_size=3, scale=8, dilation=3)
        self.layer4 = SERes2Block(channels, kernel_size=3, scale=8, dilation=4)
        self.layer5 = Conv1dReluBn(channels * 3, channels, kernel_size=1)
        self.pooling = AttentiveStatisticsPooling(channels)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(channels * 2, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim, affine=False)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        concat = torch.cat([x2, x3, x4], dim=1)
        context = self.layer5(concat)
        stats = self.pooling(context, lengths)
        stats = self.dropout(stats)
        embedding = self.fc(stats)
        embedding = self.bn(embedding)
        return F.normalize(embedding, p=2, dim=1)


class AAMSoftmaxHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, margin: float, scale: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        normalized_embeddings = F.normalize(embeddings)
        normalized_weights = F.normalize(self.weight)
        cosine = F.linear(normalized_embeddings, normalized_weights)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return self.scale * cosine, cosine, None
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = cosine * (1.0 - one_hot) + phi * one_hot
        logits = logits * self.scale
        target_cosine = cosine.gather(1, labels.view(-1, 1)).squeeze(1)
        return logits, cosine, target_cosine


class SpeakerVerificationModel(nn.Module):
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig, num_classes: int) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.feature_type = data_cfg.feature_type.lower()
        self.num_classes = num_classes
        self.wavlm: Optional[WavLMModel] = None
        self.layer_weights: Optional[nn.Parameter] = None
        if self.feature_type == "pretrained":
            self.wavlm = WavLMModel.from_pretrained(
                model_cfg.model_name,
                local_files_only=model_cfg.local_files_only,
                cache_dir=model_cfg.hf_cache_dir,
            )
            if model_cfg.use_layer_weighting:
                n_hidden = getattr(self.wavlm.config, "num_hidden_layers", len(self.wavlm.encoder.layers))
                self.layer_weights = nn.Parameter(torch.ones(n_hidden + 1))
        input_dim = model_cfg.hidden_dim if self.feature_type == "pretrained" else data_cfg.fbank_num_mel
        self.ecapa = ECAPA_TDNN_Small(
            input_dim,
            model_cfg.ecapa_channels,
            model_cfg.embedding_dim,
            dropout=model_cfg.stats_dropout,
        )
        self.classifier = AAMSoftmaxHead(model_cfg.embedding_dim, num_classes, margin=CONFIG.train.aam_margin, scale=CONFIG.train.aam_scale)
        self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        if self.wavlm is None:
            return
        for param in self.wavlm.parameters():
            param.requires_grad = False
        self.wavlm.eval()
        if not trainable:
            return
        encoder_layers = getattr(self.wavlm, "encoder", None)
        layers = getattr(encoder_layers, "layers", None)
        if layers is None:
            for param in self.wavlm.parameters():
                param.requires_grad = True
        else:
            n_layers = len(layers)
            target = self.model_cfg.unfreeze_last_n_layers
            if target <= 0 or target > n_layers:
                target = n_layers
            for layer in layers[-target:]:
                for param in layer.parameters():
                    param.requires_grad = True
        if hasattr(self.wavlm, "layer_norm"):
            for param in self.wavlm.layer_norm.parameters():
                param.requires_grad = True
        self.wavlm.train()

    def head_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = list(self.ecapa.parameters()) + [self.classifier.weight]
        if self.layer_weights is not None:
            params.append(self.layer_weights)
        return [p for p in params if p.requires_grad]

    def backbone_parameters(self) -> List[nn.Parameter]:
        if self.wavlm is None:
            return []
        return [p for p in self.wavlm.parameters() if p.requires_grad]

    def forward(self, batch: Dict[str, torch.Tensor], labels: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if self.feature_type == "pretrained":
            if self.wavlm is None:
                raise RuntimeError("WavLM backbone is not initialized.")
            wavlm_kwargs = {"output_hidden_states": self.layer_weights is not None}
            outputs = self.wavlm(
                batch["input_values"],
                attention_mask=batch.get("attention_mask"),
                **wavlm_kwargs,
            )
            if self.layer_weights is not None and outputs.hidden_states is not None:
                hidden_stack = torch.stack(outputs.hidden_states, dim=0)
                if hidden_stack.size(0) != self.layer_weights.numel():
                    hidden_stack = hidden_stack[-self.layer_weights.numel() :]
                weights = torch.softmax(self.layer_weights, dim=0)
                features = torch.einsum("l,lbsd->bsd", weights, hidden_stack)
            else:
                features = outputs.last_hidden_state
            input_lengths = batch["attention_mask"].sum(dim=1)
            if hasattr(self.wavlm, "_get_feat_extract_output_lengths"):
                lengths = self.wavlm._get_feat_extract_output_lengths(input_lengths).to(features.device)
            else:
                stride = int(np.prod(self.wavlm.config.conv_stride))
                lengths = torch.div(input_lengths + stride - 1, stride, rounding_mode="floor").to(features.device)
        else:
            features = batch["features"].to(self.classifier.weight.device)
            lengths = batch["feature_lengths"].to(features.device)
        embeddings = self.ecapa(features, lengths)
        logits, cosine, target_cosine = self.classifier(embeddings, labels)
        return logits, cosine, target_cosine, embeddings



def compute_inter_topk_penalty(
    cosine: torch.Tensor,
    target_cosine: torch.Tensor,
    labels: torch.Tensor,
    topk: int,
    margin: float,
) -> torch.Tensor:
    if topk <= 0 or cosine.size(1) <= 1:
        return torch.tensor(0.0, device=cosine.device, dtype=cosine.dtype)
    masked = cosine.clone()
    masked.scatter_(1, labels.view(-1, 1), float("-inf"))
    k = min(topk, cosine.size(1) - 1)
    topk_vals, _ = torch.topk(masked, k=k, dim=1)
    penalty = F.relu(topk_vals + margin - target_cosine.unsqueeze(1))
    return penalty.sum(dim=1).mean()


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) if torch.is_tensor(tensor) else tensor for key, tensor in batch.items()}


def train_one_epoch(
    model: SpeakerVerificationModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    train_cfg: TrainConfig,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    amp_enabled = train_cfg.use_amp
    nan_warnings = 0
    suppression_logged = False
    lr_scaled = False

    progress = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)
    for batch_data, labels, _ in progress:
        labels = labels.to(device)
        batch_data = move_batch_to_device(batch_data, device)
        optimizer.zero_grad()
        with autocast(enabled=amp_enabled):
            logits, cosine, target_cosine, _ = model(batch_data, labels)
            loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=train_cfg.label_smoothing,
            )
            if target_cosine is not None:
                penalty = compute_inter_topk_penalty(
                    cosine,
                    target_cosine,
                    labels,
                    train_cfg.inter_topk,
                    train_cfg.inter_margin,
                )
                loss = loss + penalty
        if not torch.isfinite(loss):
            nan_warnings += 1
            if nan_warnings <= train_cfg.max_nan_warnings:
                logger.warning("Non-finite loss encountered (value=%s). Skipping batch.", loss.item())
            elif not suppression_logged:
                logger.warning(
                    "Additional non-finite loss warnings suppressed for this epoch (already %d occurrences).",
                    nan_warnings,
                )
                suppression_logged = True
            optimizer.zero_grad(set_to_none=True)
            if train_cfg.nan_lr_scale < 1.0:
                for group in optimizer.param_groups:
                    new_lr = max(train_cfg.scheduler_eta_min, group["lr"] * train_cfg.nan_lr_scale)
                    if new_lr < group["lr"]:
                        group["lr"] = new_lr
                        lr_scaled = True
            if lr_scaled and nan_warnings == 1:
                logger.info("Reduced learning rates by factor %.2f to mitigate instabilities.", train_cfg.nan_lr_scale)
            if train_cfg.disable_amp_on_nan and amp_enabled:
                amp_enabled = False
                logger.warning("Disabling AMP for the remainder of this epoch due to non-finite loss.")
            continue
        scaler.scale(loss).backward()
        if train_cfg.max_grad_norm is not None:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / max(total, 1):.2f}%")
    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / total if total > 0 else 0.0
    if train_cfg.disable_amp_on_nan and train_cfg.use_amp and not amp_enabled:
        train_cfg.use_amp = False
        logger.info("AMP has been disabled for subsequent epochs due to detected instabilities.")
    return avg_loss, accuracy


def evaluate(
    model: SpeakerVerificationModel,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    log_details: bool,
    split_name: str,
) -> Dict[str, float]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_subjects: List[str] = []
    all_logits: List[List[float]] = []

    with torch.no_grad():
        for batch_data, labels, subjects in tqdm(dataloader, desc=f"Evaluating[{split_name}]", leave=False):
            labels = labels.to(device)
            batch_data = move_batch_to_device(batch_data, device)
            with autocast(enabled=use_amp):
                logits, _, _, _ = model(batch_data, labels)
                probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())
            all_subjects.extend(subjects)
            all_logits.extend(logits.cpu().tolist())

    results: Dict[str, float] = {
        "segment_acc": 0.0,
        "subject_acc": 0.0,
        "f1": 0.0,
        "auc": float("nan"),
    }

    if not all_labels:
        results["confusion_matrix"] = None
        if log_details:
            logger.info("[%s] evaluation skipped due to empty label set.", split_name)
        return results

    results["segment_acc"] = accuracy_score(all_labels, all_preds)
    results["f1"] = f1_score(all_labels, all_preds, zero_division=0)
    try:
        if len(set(all_labels)) > 1:
            results["auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        results["auc"] = float("nan")
    seg_cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    results["confusion_matrix"] = seg_cm

    subject_logits: Dict[str, List[List[float]]] = defaultdict(list)
    subject_true: Dict[str, int] = {}
    for logits, label, subject in zip(all_logits, all_labels, all_subjects):
        subject_logits[subject].append(logits)
        subject_true[subject] = label

    subject_preds: List[int] = []
    subject_labels: List[int] = []
    for subject, logits_list in subject_logits.items():
        logits_tensor = torch.tensor(logits_list, dtype=torch.float32)
        mean_logits = logits_tensor.mean(dim=0)
        mean_probs = torch.softmax(mean_logits, dim=0)
        subject_preds.append(int(torch.argmax(mean_probs).item()))
        subject_labels.append(subject_true[subject])

    if subject_labels:
        results["subject_acc"] = accuracy_score(subject_labels, subject_preds)
        results["subject_confusion_matrix"] = confusion_matrix(subject_labels, subject_preds, labels=[0, 1])

    if log_details:
        logger.info(
            "[%s] SEGMENT | correct %d/%d (acc=%.4f) | F1(seg)=%.4f | AUC(seg)=%.4f",
            split_name,
            (np.array(all_preds) == np.array(all_labels)).sum(),
            len(all_labels),
            results["segment_acc"],
            results["f1"],
            results["auc"],
        )
        logger.info("[%s] SEGMENT confusion matrix:\n%s", split_name, results["confusion_matrix"])
        if subject_labels:
            logger.info(
                "[%s] SUBJECT | correct %d/%d (acc=%.4f)",
                split_name,
                (np.array(subject_preds) == np.array(subject_labels)).sum(),
                len(subject_labels),
                results["subject_acc"],
            )
            logger.info(
                "[%s] SUBJECT confusion matrix:\n%s",
                split_name,
                results.get("subject_confusion_matrix"),
            )

    return results


def plot_training_curves(losses: List[float], metrics: List[Dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    if not losses or not metrics:
        return

    epochs = list(range(1, len(losses) + 1))
    seg_acc = [m.get("segment_acc", 0.0) for m in metrics]
    f1_scores = [m.get("f1", 0.0) for m in metrics]
    auc_scores = [m.get("auc", float("nan")) for m in metrics]
    auc_scores = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in auc_scores]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(epochs, losses, label="Train Loss")
    axes[0].set_title("Training Loss")

    axes[1].plot(epochs, seg_acc, label="Val Segment Acc")
    axes[1].set_title("Validation Segment Acc")

    axes[2].plot(epochs, f1_scores, label="Val Segment F1")
    axes[2].plot(epochs, auc_scores, label="Val Segment AUC")
    axes[2].set_title("Validation Segment F1 & AUC")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    fig.savefig(CONFIG.train.curve_path, dpi=300, bbox_inches="tight")
    logger.info("Training curves saved to %s", CONFIG.train.curve_path)


def write_epoch_history(records: List[Dict], filepath: str) -> None:
    if not records:
        return
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    df = pd.DataFrame(records)
    write_header = not os.path.exists(filepath)
    df.to_csv(filepath, mode="a", header=write_header, index=False)



def determine_total_epochs(train_cfg: TrainConfig, feature_type: str) -> int:
    if feature_type.lower() == "fbank":
        return train_cfg.fbank_epochs
    return train_cfg.pretrained_freeze_epochs + train_cfg.pretrained_finetune_epochs


def log_training_plan(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> None:
    feature_type = data_cfg.feature_type.lower()
    if feature_type == "fbank":
        logger.info(
            "Training with handcrafted Fbank features (%d mel bins, window %d ms, shift %d ms) for %d epochs (train_cfg.fbank_epochs).",
            data_cfg.fbank_num_mel,
            data_cfg.fbank_frame_length_ms,
            data_cfg.fbank_frame_shift_ms,
            train_cfg.fbank_epochs,
        )
    else:
        total_epochs = train_cfg.pretrained_freeze_epochs + train_cfg.pretrained_finetune_epochs
        logger.info(
            "Training with pretrained %s features. Stage 1: %d frozen epochs, Stage 2: %d finetune epochs (total %d epochs).",
            model_cfg.model_name,
            train_cfg.pretrained_freeze_epochs,
            train_cfg.pretrained_finetune_epochs,
            total_epochs,
        )


def epoch_stage_name(feature_type: str, epoch: int, train_cfg: TrainConfig) -> str:
    feature_type = feature_type.lower()
    if feature_type == "fbank":
        return "fbank"
    if epoch <= train_cfg.pretrained_freeze_epochs:
        return "frozen-backbone"
    return "finetune-backbone"


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
    train_dataset = AudioDataset(train_data, CONFIG.data)
    val_dataset = AudioDataset(val_data, CONFIG.data)
    test_dataset = AudioDataset(test_data, CONFIG.data)
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
    log_training_plan(CONFIG.train, CONFIG.data, CONFIG.model)
    model = SpeakerVerificationModel(CONFIG.data, CONFIG.model, num_classes).to(CONFIG.device)
    head_params = model.head_parameters()
    optimizer = optim.AdamW(
        [
            {"params": head_params, "lr": CONFIG.train.head_learning_rate},
        ],
        weight_decay=CONFIG.train.weight_decay,
    )
    total_epochs = determine_total_epochs(CONFIG.train, CONFIG.data.feature_type)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=CONFIG.train.scheduler_eta_min,
    )
    scaler = GradScaler(enabled=CONFIG.train.use_amp and CONFIG.device.type == "cuda")
    val_metric_key = CONFIG.train.val_selection_metric.lower()
    if val_metric_key not in {"segment", "subject"}:
        raise ValueError("val_selection_metric must be 'segment' or 'subject'")
    metric_field = "segment_acc" if val_metric_key == "segment" else "subject_acc"
    best_val_score = float("-inf")
    baseline_log_path = os.path.join(CONFIG.train.log_dir, "epoch_metrics.csv")

    if CONFIG.data.feature_type.lower() == "pretrained":
        model.set_backbone_trainable(False)

    if CONFIG.train.resume_from_best and os.path.exists(CONFIG.train.best_model_path):
        state_dict = torch.load(CONFIG.train.best_model_path, map_location=CONFIG.device)
        model.load_state_dict(state_dict)
        baseline = evaluate(
            model,
            val_loader,
            CONFIG.device,
            False,
            CONFIG.train.log_eval_details,
            "val-baseline",
        )
        best_val_score = baseline.get(metric_field, float("-inf"))
        logger.info(
            "Resumed from %s with baseline %s=%.4f",
            CONFIG.train.best_model_path,
            metric_field,
            best_val_score,
        )
        baseline_record = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "phase": "baseline",
            "epoch": 0,
            "train_loss": None,
            "train_acc": None,
            "val_segment_acc": baseline.get("segment_acc"),
            "val_subject_acc": baseline.get("subject_acc"),
            "val_f1": baseline.get("f1"),
            "val_auc": baseline.get("auc"),
            "lr_head": optimizer.param_groups[0]["lr"],
            "lr_backbone": None,
            "best_metric": best_val_score,
            "timestamp": datetime.now().isoformat(),
        }
        write_epoch_history([baseline_record], baseline_log_path)

    train_losses: List[float] = []
    val_metrics: List[Dict[str, float]] = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    backbone_params_added = False

    current_stage = None
    for epoch in range(1, total_epochs + 1):
        stage = epoch_stage_name(CONFIG.data.feature_type, epoch, CONFIG.train)
        if stage != current_stage:
            current_stage = stage
            if stage == "fbank":
                logger.info("Entering Fbank training stage (epochs 1-%d)", total_epochs)
            elif stage == "frozen-backbone":
                logger.info(
                    "Entering pretrained frozen-backbone stage (epochs 1-%d)",
                    CONFIG.train.pretrained_freeze_epochs,
                )
            else:
                logger.info(
                    "Entering pretrained finetuning stage (epochs %d-%d)",
                    CONFIG.train.pretrained_freeze_epochs + 1,
                    total_epochs,
                )
        logger.info("Epoch %d/%d | stage=%s", epoch, total_epochs, stage)
        if (
            CONFIG.data.feature_type.lower() == "pretrained"
            and not backbone_params_added
            and epoch > CONFIG.train.pretrained_freeze_epochs
        ):
            model.set_backbone_trainable(True)
            backbone_params = model.backbone_parameters()
            if backbone_params:
                optimizer.add_param_group({"params": backbone_params, "lr": CONFIG.train.backbone_learning_rate})
                scheduler.base_lrs.append(CONFIG.train.backbone_learning_rate)
                backbone_params_added = True

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            CONFIG.device,
            CONFIG.train,
        )
        train_losses.append(train_loss)
        logger.info("Train | loss=%.4f | acc=%.4f", train_loss, train_acc)

        val_result = evaluate(
            model,
            val_loader,
            CONFIG.device,
            False,
            CONFIG.train.log_eval_details,
            f"val-epoch{epoch}",
        )
        val_metrics.append(val_result)
        logger.info(
            "Val | segment_acc=%.4f | subject_acc=%.4f | f1=%.4f | auc=%.4f",
            val_result.get("segment_acc", 0.0),
            val_result.get("subject_acc", 0.0),
            val_result.get("f1", 0.0),
            val_result.get("auc", float("nan")),
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
        lr_head = optimizer.param_groups[0]["lr"]
        lr_backbone = None
        if backbone_params_added:
            lr_backbone = optimizer.param_groups[-1]["lr"]
        epoch_record = {
            "run_id": run_id,
            "phase": "epoch",
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_segment_acc": val_result.get("segment_acc"),
            "val_subject_acc": val_result.get("subject_acc"),
            "val_f1": val_result.get("f1"),
            "val_auc": val_result.get("auc"),
            "lr_head": lr_head,
            "lr_backbone": lr_backbone,
            "best_metric": best_val_score,
            "timestamp": datetime.now().isoformat(),
        }
        write_epoch_history([epoch_record], baseline_log_path)

    logger.info("Evaluating on test set")
    if os.path.exists(CONFIG.train.best_model_path):
        model.load_state_dict(torch.load(CONFIG.train.best_model_path, map_location=CONFIG.device))
    test_result = evaluate(
        model,
        test_loader,
        CONFIG.device,
        False,
        CONFIG.train.log_eval_details,
        "test",
    )
    logger.info(
        "Test | segment_acc=%.4f | subject_acc=%.4f | f1=%.4f | auc=%.4f",
        test_result.get("segment_acc", 0.0),
        test_result.get("subject_acc", 0.0),
        test_result.get("f1", 0.0),
        test_result.get("auc", float("nan")),
    )
    logger.info("Confusion matrix:\n%s", test_result.get("confusion_matrix"))
    test_record = {
        "run_id": run_id,
        "phase": "test",
        "epoch": "test",
        "train_loss": None,
        "train_acc": None,
        "val_segment_acc": None,
        "val_subject_acc": None,
        "val_f1": None,
        "val_auc": None,
        "lr_head": optimizer.param_groups[0]["lr"],
        "lr_backbone": optimizer.param_groups[-1]["lr"] if backbone_params_added else None,
        "best_metric": best_val_score,
        "test_segment_acc": test_result.get("segment_acc"),
        "test_subject_acc": test_result.get("subject_acc"),
        "test_f1": test_result.get("f1"),
        "test_auc": test_result.get("auc"),
        "timestamp": datetime.now().isoformat(),
    }
    write_epoch_history([test_record], baseline_log_path)
    if CONFIG.train.plot_training_curves:
        plot_training_curves(train_losses, val_metrics)
    logger.info("Experiment complete")


if __name__ == "__main__":
    main()

