import logging
import os
import warnings
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
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
    tag = model_name.replace("/", "_").replace(":", "_")
    return tag


@dataclass
class DataConfig:
    data_dir: str = "./audio_lanzhou_2015"
    sample_rate: int = 16000
    segment_duration: int = 10
    overlap_ratio: float = 0.2
    max_segments_per_subject: Optional[int] = None
    apply_augmentation: bool = True
    normalize_amplitude: bool = True
    apply_silence_trim: bool = True
    silence_frame_ms: int = 25
    silence_hop_ms: int = 10
    silence_energy_threshold: float = 1e-4
    apply_median_filter: bool = True
    median_filter_kernel: int = 5
    generate_hamming_frames: bool = False
    frame_length_ms: int = 25
    frame_overlap_ms: int = 15


@dataclass
class ModelConfig:
    model_name: str = "microsoft/wavlm-base-plus"
    hidden_dim: int = 768
    unfreeze_last_n_layers: int = 0
    hf_cache_dir: str = "./hf_cache"
    local_files_only: bool = True
    force_offline: bool = True
    use_layer_weighting: bool = True

    @property
    def tag(self) -> str:
        return _safe_model_tag(self.model_name)


@dataclass
class TrainConfig:
    batch_size: int = 128
    num_workers: int = 4
    persistent_workers: bool = True
    num_epochs: int = 100
    head_learning_rate: float = 5e-5
    backbone_learning_rate: float = 3e-5
    weight_decay: float = 1e-5
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


# Utility function

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


# Audio preprocessing

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
    padded = torch.nn.functional.pad(waveform.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)
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


# Dataset

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
        model_cfg: ModelConfig,
        split: str,
    ) -> None:
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.split = split
        self.segment_length = data_cfg.sample_rate * data_cfg.segment_duration
        self.hop_length = max(1, int(self.segment_length * (1 - data_cfg.overlap_ratio)))
        self.apply_augmentation = data_cfg.apply_augmentation and split == "train"
        self.samples: List[Dict] = []
        subject_counts = Counter()
        self.stft_window = torch.hann_window(400, periodic=True, dtype=torch.float32)
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

    def _crop_segment(self, waveform: torch.Tensor, offset: int) -> torch.Tensor:
        target = self.segment_length
        if waveform.size(0) <= target:
            return torch.nn.functional.pad(waveform, (0, target - waveform.size(0)))
        jitter = 0
        if self.apply_augmentation:
            half_hop = max(1, self.hop_length // 2)
            jitter = random.randint(-half_hop, half_hop)
        offset = int(min(max(offset + jitter, 0), waveform.size(0) - target))
        return waveform[offset : offset + target]

    def _time_stretch(self, segment: torch.Tensor) -> torch.Tensor:
        rate = random.uniform(0.9, 1.1)
        new_sr = max(1000, int(self.data_cfg.sample_rate * rate))
        stretched = torchaudio.functional.resample(
            segment.unsqueeze(0),
            self.data_cfg.sample_rate,
            new_sr,
        ).squeeze(0)
        if stretched.size(0) > segment.size(0):
            start = random.randint(0, stretched.size(0) - segment.size(0))
            stretched = stretched[start : start + segment.size(0)]
        else:
            stretched = torch.nn.functional.pad(stretched, (0, segment.size(0) - stretched.size(0)))
        return stretched

    def _random_eq(self, segment: torch.Tensor) -> torch.Tensor:
        gain_db = random.uniform(-3.0, 3.0)
        center_freq = random.uniform(200.0, 3500.0)
        q = random.uniform(0.5, 1.5)
        eq = torchaudio.functional.equalizer_biquad(
            segment.unsqueeze(0),
            self.data_cfg.sample_rate,
            center_freq,
            gain_db,
            q=q,
        )
        return eq.squeeze(0)

    def _bandpass_noise(self, segment: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(segment)
        low_freq = random.uniform(100.0, 1000.0)
        high_freq = random.uniform(1500.0, 6000.0)
        if high_freq <= low_freq:
            high_freq = low_freq + 500.0
        nyquist = self.data_cfg.sample_rate / 2.0
        high_freq = min(high_freq, nyquist - 100.0)
        low_freq = max(50.0, min(low_freq, high_freq - 100.0))
        filtered = torchaudio.functional.bandpass_biquad(
            noise.unsqueeze(0),
            self.data_cfg.sample_rate,
            center_freq=(low_freq + high_freq) / 2.0,
            Q=1.0,
        ).squeeze(0)
        noise_level = random.uniform(0.001, 0.01)
        return segment + filtered * noise_level

    def _specaugment(self, segment: torch.Tensor) -> torch.Tensor:
        n_fft = 400
        hop = 160
        win = 400
        window = self.stft_window.to(segment.device)
        spec = torch.stft(
            segment,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True,
        )
        spec = spec.clone()
        if spec.size(1) > 0:
            t_mask = random.randint(0, max(0, spec.size(1) // 6))
            t_start = random.randint(0, max(0, spec.size(1) - t_mask)) if t_mask > 0 else 0
            if t_mask > 0:
                spec[:, t_start : t_start + t_mask] = 0
        if spec.size(0) > 0:
            f_mask = random.randint(0, max(0, spec.size(0) // 8))
            f_start = random.randint(0, max(0, spec.size(0) - f_mask)) if f_mask > 0 else 0
            if f_mask > 0:
                spec[f_start : f_start + f_mask, :] = 0
        augmented = torch.istft(
            spec,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            length=segment.size(0),
        )
        return augmented

    def _augment(self, segment: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            segment = self._time_stretch(segment)
        if random.random() < 0.7:
            segment = self._random_eq(segment)
        if random.random() < 0.5:
            segment = self._bandpass_noise(segment)
        if random.random() < 0.5:
            segment = self._specaugment(segment)
        gain = random.uniform(0.85, 1.15)
        segment = segment * gain
        segment = segment.clamp(-1.0, 1.0)
        return segment

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        waveform = self._load_waveform(sample["path"])
        segment = self._crop_segment(waveform, sample["offset"]).float()
        if self.apply_augmentation:
            segment = self._augment(segment)
        return segment, sample["label"], sample["subject"]


def collate_fn(
    batch: List[Tuple[torch.Tensor, int, str]],
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
):
    segments, labels, subjects = zip(*batch)
    extractor = get_feature_extractor(model_cfg)
    segments_np = [seg.cpu().numpy() for seg in segments]
    processed = extractor(
        segments_np,
        sampling_rate=data_cfg.sample_rate,
        padding=True,
        return_tensors="pt",
    )
    return (
        processed.input_values,
        processed.attention_mask.long(),
        torch.tensor(labels, dtype=torch.long),
        list(subjects),
    )


# Model definition


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        float_mask = padding_mask.unsqueeze(-1).type_as(hidden_states)
        attn_logits = self.attention(hidden_states).masked_fill(
            padding_mask.unsqueeze(-1) == 0,
            float("-inf"),
        )
        attn_weights = torch.softmax(attn_logits, dim=1)
        mean = torch.sum(hidden_states * attn_weights * float_mask, dim=1)
        variance = torch.sum(
            ((hidden_states - mean.unsqueeze(1)) ** 2) * attn_weights * float_mask,
            dim=1,
        )
        std = torch.sqrt(torch.clamp(variance, min=1e-8))
        return torch.cat([mean, std], dim=-1)


class DepressionClassifier(nn.Module):
    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.wavlm = WavLMModel.from_pretrained(
            model_cfg.model_name,
            local_files_only=model_cfg.local_files_only,
            cache_dir=model_cfg.hf_cache_dir,
        )
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False
        if model_cfg.unfreeze_last_n_layers > 0:
            self._unfreeze_last_layers(model_cfg.unfreeze_last_n_layers)
        self.wavlm_trainable = any(param.requires_grad for param in self.wavlm.parameters())

        self.use_layer_weighting = model_cfg.use_layer_weighting
        self.layer_weights: Optional[nn.Parameter] = None
        if self.use_layer_weighting:
            n_hidden = getattr(
                self.wavlm.config,
                "num_hidden_layers",
                len(self.wavlm.encoder.layers),
            )
            self.layer_weights = nn.Parameter(torch.ones(n_hidden + 1))

        self.pooling = AttentiveStatisticsPooling(model_cfg.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(model_cfg.hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
        self._trainable = [p for p in self.parameters() if p.requires_grad]

    def _unfreeze_last_layers(self, n_layers: int) -> None:
        encoder_layers = self.wavlm.encoder.layers
        for layer in encoder_layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.wavlm.layerdrop = 0.0
        if hasattr(self.wavlm, "gradient_checkpointing_enable"):
            self.wavlm.gradient_checkpointing_enable()

    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        return self._trainable

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        wavlm_kwargs = {"output_hidden_states": self.use_layer_weighting}
        if self.wavlm_trainable:
            outputs = self.wavlm(input_values, attention_mask=attention_mask, **wavlm_kwargs)
        else:
            with torch.no_grad():
                outputs = self.wavlm(input_values, attention_mask=attention_mask, **wavlm_kwargs)

        if self.use_layer_weighting and outputs.hidden_states is not None and self.layer_weights is not None:
            hidden_stack = torch.stack(outputs.hidden_states, dim=0)
            if hidden_stack.size(0) != self.layer_weights.numel():
                hidden_stack = hidden_stack[-self.layer_weights.numel() :]
            weights = torch.softmax(self.layer_weights, dim=0)
            hidden_states = torch.einsum("l,lbsd->bsd", weights, hidden_stack)
        else:
            hidden_states = outputs.last_hidden_state
        input_lengths = attention_mask.sum(dim=-1)
        if hasattr(self.wavlm, "_get_feat_extract_output_lengths"):
            feat_lengths = self.wavlm._get_feat_extract_output_lengths(input_lengths).to(hidden_states.device)
        else:
            stride = int(np.prod(self.wavlm.config.conv_stride))
            feat_lengths = torch.div(
                input_lengths + stride - 1,
                stride,
                rounding_mode="floor",
            ).to(hidden_states.device)
        max_len = hidden_states.size(1)
        frame_index = torch.arange(max_len, device=hidden_states.device).unsqueeze(0)
        padding_mask = frame_index < feat_lengths.unsqueeze(1)
        embeddings = self.pooling(hidden_states, padding_mask)
        return self.classifier(embeddings)


# Training / evaluation

def train_one_epoch(
    model: DepressionClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
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
    for input_values, attention_mask, labels, _ in progress:
        input_values = input_values.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast(enabled=amp_enabled):
            logits = model(input_values, attention_mask)
            loss = criterion(logits, labels)
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
            clip_grad_norm_(model.trainable_parameters, train_cfg.max_grad_norm)
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
    model: DepressionClassifier,
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
        for input_values, attention_mask, labels, subjects in tqdm(
            dataloader,
            desc=f"Evaluating[{split_name}]",
            leave=False,
        ):
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            with autocast(enabled=use_amp):
                logits = model(input_values, attention_mask)
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
    subject_prob_values: List[float] = []
    for subject, logits_list in subject_logits.items():
        logits_tensor = torch.tensor(logits_list, dtype=torch.float32)
        mean_logits = logits_tensor.mean(dim=0)
        mean_probs = torch.softmax(mean_logits, dim=0)
        subject_preds.append(int(torch.argmax(mean_probs).item()))
        subject_prob_values.append(float(mean_probs[1].item()))
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


# Main experiment workflow

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
    get_feature_extractor(CONFIG.model)
    train_dataset = AudioDataset(train_data, CONFIG.data, CONFIG.model, split="train")
    val_dataset = AudioDataset(val_data, CONFIG.data, CONFIG.model, split="val")
    test_dataset = AudioDataset(test_data, CONFIG.data, CONFIG.model, split="test")
    if not len(train_dataset):
        raise RuntimeError("Training dataset is empty.")
    sampler = None
    if train_dataset.sample_weights.numel() > 0:
        sampler = WeightedRandomSampler(
            train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )

    def _collate(batch):
        return collate_fn(batch, CONFIG.model, CONFIG.data)

    common_loader_kwargs = dict(
        num_workers=CONFIG.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
        persistent_workers=CONFIG.train.persistent_workers and CONFIG.train.num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.train.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.train.batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG.train.batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    epoch_history: List[Dict] = []
    model = DepressionClassifier(CONFIG.model).to(CONFIG.device)
    if not model.trainable_parameters:
        raise RuntimeError("No trainable parameters detected. Adjust unfreeze settings.")
    criterion = nn.CrossEntropyLoss()
    head_params = [
        p
        for p in list(model.classifier.parameters()) + list(model.pooling.parameters())
        if p.requires_grad
    ]
    if model.use_layer_weighting and model.layer_weights is not None and model.layer_weights.requires_grad:
        head_params.append(model.layer_weights)
    backbone_params = [p for p in model.wavlm.parameters() if p.requires_grad]
    optimizer_groups = []
    if head_params:
        optimizer_groups.append({"params": head_params, "lr": CONFIG.train.head_learning_rate})
    if backbone_params:
        optimizer_groups.append({"params": backbone_params, "lr": CONFIG.train.backbone_learning_rate})
    optimizer = optim.AdamW(optimizer_groups, weight_decay=CONFIG.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG.train.num_epochs,
        eta_min=CONFIG.train.scheduler_eta_min,
    )
    scaler = GradScaler(enabled=CONFIG.train.use_amp and CONFIG.device.type == "cuda")
    val_metric_key = CONFIG.train.val_selection_metric.lower()
    if val_metric_key not in {"segment", "subject"}:
        raise ValueError("val_selection_metric must be 'segment' or 'subject'")
    metric_field = "segment_acc" if val_metric_key == "segment" else "subject_acc"
    best_val_score = float("-inf")
    baseline_log_path = os.path.join(CONFIG.train.log_dir, "epoch_metrics.csv")
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
            "run_id": run_id,
            "phase": "baseline",
            "epoch": 0,
            "train_loss": None,
            "train_acc": None,
            "val_segment_acc": baseline.get("segment_acc"),
            "val_subject_acc": baseline.get("subject_acc"),
            "val_f1": baseline.get("f1"),
            "val_auc": baseline.get("auc"),
            "lr_head": optimizer.param_groups[0]["lr"],
            "lr_backbone": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else None,
            "best_metric": best_val_score,
            "timestamp": datetime.now().isoformat(),
        }
        epoch_history.append(baseline_record)
        write_epoch_history([baseline_record], baseline_log_path)
    train_losses: List[float] = []
    val_metrics: List[Dict[str, float]] = []
    for epoch in range(1, CONFIG.train.num_epochs + 1):
        logger.info("Epoch %d/%d", epoch, CONFIG.train.num_epochs)
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
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
        lr_backbone = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else None
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
        epoch_history.append(epoch_record)
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
        "lr_backbone": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else None,
        "best_metric": best_val_score,
        "test_segment_acc": test_result.get("segment_acc"),
        "test_subject_acc": test_result.get("subject_acc"),
        "test_f1": test_result.get("f1"),
        "test_auc": test_result.get("auc"),
        "timestamp": datetime.now().isoformat(),
    }
    epoch_history.append(test_record)
    write_epoch_history([test_record], baseline_log_path)
    if CONFIG.train.plot_training_curves:
        plot_training_curves(train_losses, val_metrics)
    logger.info("Experiment complete")


if __name__ == "__main__":
    main()
