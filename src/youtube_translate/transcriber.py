"""Whisper 语音转写模块"""
import datetime
import logging

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
    """
    英文语音转文字，使用词级时间戳智能分割字幕

    优势：
    - 词级时间精度
    - 按标点符号智能分句
    - 避免在数字中间断开（如 3.14）
    - 控制字幕长度，避免单条过长
    """
    device, compute_type = _get_whisper_device()

    model = WhisperModel(
        model_name, device=device, compute_type=compute_type,
        download_root="models/whisper", local_files_only=True,
    )
    logging.info("Whisper 模型已加载")

    # 使用 word_timestamps 获取词级时间精度
    segments, info = model.transcribe(
        audio=audio_path, language=language,
        word_timestamps=True,  # 关键：获取每个词的时间戳
        initial_prompt=None, log_progress=True,
    )

    # 按标点符号和长度智能分割字幕
    subs = []
    current_sub = None
    sub_index = 1

    # 句末标点
    sentence_end_punctuation = {".", "!", "?", ";", "…"}

    for segment in segments:
        if not segment.words:
            continue

        for word in segment.words:
            word_text = word.word.strip()
            if not word_text:
                continue

            # 创建新字幕
            if current_sub is None:
                current_sub = srt.Subtitle(
                    index=sub_index,
                    start=datetime.timedelta(seconds=word.start),
                    end=datetime.timedelta(seconds=word.end),
                    content=word_text
                )
            else:
                # 追加到当前字幕
                current_sub.end = datetime.timedelta(seconds=word.end)
                current_sub.content += " " + word_text

            # 判断是否应该结束当前字幕
            is_sentence_end = (
                len(word_text) > 0 and
                word_text[-1] in sentence_end_punctuation and
                # 避免在数字中间断开（如 3.14）
                not (len(word_text) > 1 and word_text[-2].isdigit())
            )

            # 字幕过长也强制分割（超过 100 字符）
            is_too_long = len(current_sub.content) > 100

            if is_sentence_end or is_too_long:
                subs.append(current_sub)
                current_sub = None
                sub_index += 1

    # 添加最后一条字幕
    if current_sub is not None:
        subs.append(current_sub)

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

    # 中文使用 word_timestamps 效果可能不如 segment
    # 因为中文分词不如英文准确
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
