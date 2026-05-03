"""Whisper 语音转写模块"""
import datetime
import logging

import srt
import torch
import whisper


def _get_whisper_device():
    """获取 Whisper 模型的最优设备"""
    if torch.cuda.is_available():
        return "cuda"
    # Intel Mac 使用 CPU
    return "cpu"


def transcribe_audio_en(
    audio_path: str,
    model_name: str,
    language: str,
    output_srt_path: str,
) -> bool:
    """
    英文语音转文字，使用原生 OpenAI Whisper

    相比 faster-whisper，原生 Whisper 更稳定，
    在 Intel Mac 上不会出现段错误
    """
    device = _get_whisper_device()

    logging.info(f"加载 Whisper 模型: {model_name}, 设备: {device}")
    model = whisper.load_model(model_name, device=device)
    logging.info("Whisper 模型已加载")

    # 转写音频（显示进度条）
    result = model.transcribe(
        audio=audio_path,
        language=language,
        verbose=False,  # 关闭字幕输出
    )

    # 转换为 SRT 格式
    subs = []
    for i, segment in enumerate(result["segments"], 1):
        sub = srt.Subtitle(
            index=i,
            start=datetime.timedelta(seconds=segment["start"]),
            end=datetime.timedelta(seconds=segment["end"]),
            content=segment["text"].strip(),
        )
        subs.append(sub)

    logging.info("转写完成，共 %d 条字幕", len(subs))

    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
    return True


def transcribe_audio_zh(audio_path: str, model_name: str, output_srt_path: str) -> None:
    """中文语音转文字"""
    device = _get_whisper_device()

    logging.info(f"加载 Whisper 模型: {model_name}, 设备: {device}")
    model = whisper.load_model(model_name, device=device)

    result = model.transcribe(
        audio=audio_path,
        language="zh",
        initial_prompt="简体",
        verbose=False,  # 关闭字幕输出
    )

    subs = []
    for i, segment in enumerate(result["segments"], 1):
        sub = srt.Subtitle(
            index=i,
            start=datetime.timedelta(seconds=segment["start"]),
            end=datetime.timedelta(seconds=segment["end"]),
            content=segment["text"].strip(),
        )
        subs.append(sub)

    logging.info("中文转写完成，共 %d 条字幕", len(subs))

    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
