"""Whisper 语音转写模块"""
import datetime
import logging
from typing import Optional

import srt
import torch
from faster_whisper import WhisperModel


def _get_whisper_device():
    """获取 Whisper 模型的最优设备"""
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


def transcribe_audio_en(
    audio_path: str,
    model_name: str,
    language: str,
    output_srt_path: str,
) -> bool:
    """英文语音转文字，直接输出句级字幕"""
    device, compute_type = _get_whisper_device()

    model = WhisperModel(
        model_name, device=device, compute_type=compute_type,
        download_root="models/whisper", local_files_only=True,
    )
    logging.info("Whisper 模型已加载")

    segments, info = model.transcribe(
        audio=audio_path, language=language,
        initial_prompt=None, log_progress=True,
    )

    # faster_whisper 直接输出句级 segment，无需手动拼词
    subs = []
    for i, segment in enumerate(segments, 1):
        sub = srt.Subtitle(
            index=i,
            start=datetime.timedelta(seconds=segment.start),
            end=datetime.timedelta(seconds=segment.end),
            content=segment.text.strip(),
        )
        subs.append(sub)

    logging.info("转写完成，共 %d 条字幕", len(subs))

    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
    return True


def transcribe_audio_zh(audio_path: str, model_name: str, output_srt_path: str) -> None:
    """中文语音转文字"""
    device, compute_type = _get_whisper_device()
    model = WhisperModel(
        model_name, device=device, compute_type=compute_type,
        download_root="models/whisper", local_files_only=True,
    )

    segments, _ = model.transcribe(
        audio=audio_path, language="zh",
        initial_prompt="简体", log_progress=True,
    )

    subs = []
    for i, segment in enumerate(segments, 1):
        sub = srt.Subtitle(
            index=i,
            start=datetime.timedelta(seconds=segment.start),
            end=datetime.timedelta(seconds=segment.end),
            content=segment.text.strip(),
        )
        subs.append(sub)

    logging.info("中文转写完成，共 %d 条字幕", len(subs))

    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
