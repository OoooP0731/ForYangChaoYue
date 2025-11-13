"""Gradio interface for WavLM-Large depression screening.

This application loads a fine-tuned wavlm-large classifier (trained via
``train.py``) and exposes a browser interface where users can upload a
checkpoint (`best_model.pt`) and an audio clip (.wav). The app segments the
audio the same way as training (7 s with 20% overlap), feeds each segment
through the model, and aggregates segment probabilities to predict whether the
speaker exhibits depressive speech patterns.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_removed_sys_paths: List[Tuple[int, str]] = []
for idx in reversed(range(len(sys.path))):
    try:
        if Path(sys.path[idx]).resolve() == _THIS_DIR:
            _removed_sys_paths.append((idx, sys.path.pop(idx)))
    except Exception:
        continue
sys.modules.pop("gradio", None)
try:
    gr = importlib.import_module("gradio")  # type: ignore
finally:
    for insert_idx, path_value in sorted(_removed_sys_paths, key=lambda item: item[0]):
        sys.path.insert(insert_idx, path_value)
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F

from train import (
    DataConfig,
    ModelConfig,
    Wav2Vec2Classifier,
    get_feature_extractor,
    preprocess_waveform,
)

# ---------------------------------------------------------------------------
# Global configuration and model state
# ---------------------------------------------------------------------------

DATA_CFG = DataConfig()
MODEL_CFG = ModelConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL: Optional[Wav2Vec2Classifier] = None
_CURRENT_CHECKPOINT: Optional[Path] = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_path(file_obj: Optional[object]) -> Optional[Path]:
    """Extract a Path from a Gradio file object or raw path string."""

    if file_obj is None:
        return None
    if isinstance(file_obj, (str, os.PathLike)):
        return Path(file_obj)
    if hasattr(file_obj, "name"):
        return Path(getattr(file_obj, "name"))
    if isinstance(file_obj, dict) and "name" in file_obj:
        return Path(file_obj["name"])  # type: ignore[index]
    raise ValueError("无法识别的文件对象。")


def _default_checkpoint() -> Optional[Path]:
    """Return the first existing checkpoint candidate on disk."""

    candidates = [
        Path("best_model_wavlm_lagre.pt"),
        Path("best_model.pt"),
        Path("best_segment_model.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def ensure_model_loaded(checkpoint: Path) -> str:
    """Load model weights from ``checkpoint`` if needed and return a status message."""

    global _MODEL, _CURRENT_CHECKPOINT

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint}")

    if _MODEL is None:
        _MODEL = Wav2Vec2Classifier(MODEL_CFG, num_classes=2).to(DEVICE)

    if _CURRENT_CHECKPOINT == checkpoint:
        return f"已加载模型：{checkpoint.name}"

    state = torch.load(checkpoint, map_location=DEVICE)
    missing, unexpected = _MODEL.load_state_dict(state, strict=False)
    _MODEL.eval()
    _CURRENT_CHECKPOINT = checkpoint

    extra_notes = []
    if missing:
        extra_notes.append(f"缺失权重键: {missing}")
    if unexpected:
        extra_notes.append(f"存在未使用的权重键: {unexpected}")

    note_text = f" （{'；'.join(extra_notes)}）" if extra_notes else ""
    return f"成功加载模型：{checkpoint.name}{note_text}"


def _segment_audio(waveform: torch.Tensor) -> Tuple[List[np.ndarray], List[float], float]:
    """Split waveform into overlapping segments used during training."""

    segment_length = DATA_CFG.sample_rate * DATA_CFG.segment_duration
    hop_length = max(1, int(segment_length * (1 - DATA_CFG.overlap_ratio)))
    total_samples = waveform.numel()
    total_duration = total_samples / DATA_CFG.sample_rate if total_samples else 0.0

    if total_samples == 0:
        return [], [], total_duration

    offsets = []
    if total_samples <= segment_length:
        offsets = [0]
    else:
        offsets = list(range(0, total_samples - segment_length + 1, hop_length))
        last_offset = total_samples - segment_length
        if offsets[-1] != last_offset:
            offsets.append(last_offset)

    segments: List[np.ndarray] = []
    offsets_seconds: List[float] = []
    for start in offsets:
        end = start + segment_length
        segment = waveform[start:end]
        if segment.size(0) < segment_length:
            segment = F.pad(segment, (0, segment_length - segment.size(0)))
        segments.append(segment.cpu().numpy().astype(np.float32))
        offsets_seconds.append(start / DATA_CFG.sample_rate)

    return segments, offsets_seconds, total_duration


def prepare_segments(audio_path: str) -> Tuple[List[np.ndarray], List[float], float]:
    """Load an audio file, apply preprocessing, and return segmented numpy arrays."""

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    if sample_rate != DATA_CFG.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform.unsqueeze(0), sample_rate, DATA_CFG.sample_rate
        ).squeeze(0)

    waveform = preprocess_waveform(waveform, DATA_CFG)
    return _segment_audio(waveform)


def run_inference(audio_path: str, checkpoint_state: Optional[str]) -> Tuple[str, Optional[pd.DataFrame]]:
    """Execute inference on the provided audio path."""

    if not audio_path:
        return "请上传音频文件。", None

    checkpoint = Path(checkpoint_state) if checkpoint_state else _CURRENT_CHECKPOINT
    if checkpoint is None:
        default_ckpt = _default_checkpoint()
        if default_ckpt is None:
            return "未找到模型，请先上传 best_model.pt。", None
        checkpoint = default_ckpt

    try:
        status_msg = ensure_model_loaded(checkpoint)
    except Exception as exc:
        return f"模型加载失败：{exc}", None

    segments, offsets_seconds, total_duration = prepare_segments(audio_path)
    if not segments:
        return "音频内容为空或预处理后无有效片段。", None

    extractor = get_feature_extractor(MODEL_CFG)
    processed = extractor(
        segments,
        sampling_rate=DATA_CFG.sample_rate,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    batch = {"input_values": processed.input_values.to(DEVICE)}
    if processed.get("attention_mask") is not None:
        batch["attention_mask"] = processed.attention_mask.to(DEVICE)

    assert _MODEL is not None
    with torch.no_grad():
        logits, _ = _MODEL(batch)
        probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    subject_probability = float(probabilities.mean())
    threshold = 0.5
    prediction = "MDD (抑郁倾向)" if subject_probability >= threshold else "HC (健康对照)"

    records = []
    for idx, (offset, prob) in enumerate(zip(offsets_seconds, probabilities), start=1):
        records.append(
            {
                "Segment": idx,
                "Start (s)": round(offset, 2),
                "End (s)": round(min(offset + DATA_CFG.segment_duration, total_duration), 2),
                "MDD Probability": round(float(prob), 4),
            }
        )
    df = pd.DataFrame(records)

    summary = (
        f"{status_msg}\n\n"
        f"**判定结果：{prediction}**\n\n"
        f"- MDD 概率：{subject_probability:.3f}\n"
        f"- 使用片段数量：{len(segments)}\n"
        f"- 音频时长：{total_duration:.2f} 秒\n"
        f"- 判定阈值：{threshold:.2f}"
    )
    return summary, df


def handle_checkpoint_upload(file_obj: Optional[object], current_state: Optional[str]) -> Tuple[Optional[str], str]:
    """Process checkpoint uploads and update internal model state."""

    candidate = _to_path(file_obj) or (Path(current_state) if current_state else None)
    if candidate is None:
        candidate = _default_checkpoint()
        if candidate is None:
            return current_state, "请上传 .pt 模型文件后再加载。"

    try:
        message = ensure_model_loaded(candidate)
        return str(candidate), message
    except Exception as exc:
        return current_state, f"模型加载失败：{exc}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DEFAULT_CKPT = _default_checkpoint()
INITIAL_STATUS = "尚未加载模型，请先确认目录下存在 best_model_wavlm_lagre.pt。"
if DEFAULT_CKPT is not None:
    try:
        INITIAL_STATUS = ensure_model_loaded(DEFAULT_CKPT)
    except Exception as exc:  # pragma: no cover - startup feedback
        INITIAL_STATUS = f"默认模型加载失败：{exc}"

with gr.Blocks(title="WavLM-Large 抑郁倾向筛查") as demo:
    gr.Markdown(
        """
        # WavLM-Large 语音抑郁筛查

        * 系统会自动加载同目录下的 `best_model_wavlm_lagre.pt`（若存在）。
        * 如需测试其他权重，可选择新的 `.pt` 文件并点击“加载模型”。
        * 上传待评估的 `.wav` 语音后点击“开始检测”，即可获得抑郁概率与逐片段结果。正类（Positive Class）定义为 **MDD**。
        """
    )

    checkpoint_state = gr.State(value=str(DEFAULT_CKPT) if DEFAULT_CKPT else None)

    with gr.Row():
        checkpoint_input = gr.File(label="上传/选择模型权重 (.pt)", file_types=[".pt"], type="filepath")
        load_button = gr.Button("加载模型", variant="primary")

    status_box = gr.Markdown(INITIAL_STATUS)

    audio_input = gr.Audio(label="上传待检测的语音 (.wav)", type="filepath")
    detect_button = gr.Button("开始检测", variant="primary")

    result_output = gr.Markdown()
    segment_output = gr.Dataframe(headers=["Segment", "Start (s)", "End (s)", "MDD Probability"], wrap=True)

    load_button.click(
        handle_checkpoint_upload,
        inputs=[checkpoint_input, checkpoint_state],
        outputs=[checkpoint_state, status_box],
    )

    detect_button.click(
        run_inference,
        inputs=[audio_input, checkpoint_state],
        outputs=[result_output, segment_output],
    )


if __name__ == "__main__":
    demo.launch(share=True)
