"""Whisper 语音转写模块"""
import copy
import datetime
import logging
import math
import struct
import wave
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
    """英文语音转文字"""
    not_silence_threshold_db = -30
    end_interpunction = ["…", ".", "!", "?", ";"]
    number_characters = "0123456789"

    initial_prompt = "简体" if language == "zh" else None
    device, compute_type = _get_whisper_device()

    model = WhisperModel(
        model_name, device=device, compute_type=compute_type,
        download_root="models/whisper", local_files_only=True,
    )
    logging.info("Whisper 模型已加载")

    segments, info = model.transcribe(
        audio=audio_path, language=language,
        word_timestamps=True, initial_prompt=initial_prompt, log_progress=True,
    )

    index = 1
    subs = []
    subtitle = None
    segments_list = list(segments)

    for segment in segments_list:
        for word in segment.words:
            if subtitle is None:
                subtitle = srt.Subtitle(index, datetime.timedelta(seconds=word.start), datetime.timedelta(seconds=word.end), "")

            final_word = word.word.strip()
            subtitle.end = datetime.timedelta(seconds=word.end)

            is_sentence_end = (
                final_word[-1] in end_interpunction
                and not (len(final_word) > 1 and final_word[-2] in number_characters)
            )

            if is_sentence_end:
                subtitle.content += " " + final_word
                subs.append(subtitle)
                index += 1
                subtitle = None
            else:
                if subtitle.content == "":
                    subtitle.content = final_word
                elif final_word[0] == ".":
                    subtitle.content += final_word
                elif len(subtitle.content) > 0 and subtitle.content[-1] == "." and final_word[0] in number_characters:
                    subtitle.content += final_word
                else:
                    subtitle.content += " " + final_word

    if subtitle is not None:
        subs.append(subtitle)

    logging.info("转写完成")

    # 校准字幕开头时间
    audio_wav = wave.open(audio_path, "rb")
    frame_rate = audio_wav.getframerate()
    not_silence_threshold = math.pow(10, not_silence_threshold_db / 20)

    for sub in subs:
        start_time = sub.start.total_seconds()
        start_frame = int(start_time * frame_rate)
        end_time = sub.end.total_seconds()
        end_frame = int(end_time * frame_rate)

        new_start_time = start_time
        audio_wav.setpos(start_frame)
        read_frames = end_frame - start_frame

        for i in range(read_frames):
            frame = audio_wav.readframes(1)
            if not frame:
                break
            samples = struct.iter_unpack("<h", frame)
            sample_volumes = [abs(s[0]) / 32768 for s in samples]
            if max(sample_volumes) > not_silence_threshold:
                new_start_time = start_time + i / frame_rate
                break

        sub.start = datetime.timedelta(seconds=new_start_time)

    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
    return True


def transcribe_audio_zh(audio_path: str, model_name: str, output_srt_path: str) -> None:
    """中文语音转文字"""
    end_interpunction = ["。", "！", "？", "…", "；", "，", "、", ",", ".", "!", "?", ";"]
    en_num_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    device, compute_type = _get_whisper_device()
    model = WhisperModel(
        model_name, device=device, compute_type=compute_type,
        download_root="models/whisper", local_files_only=True,
    )
    segments, _ = model.transcribe(audio=audio_path, language="zh", word_timestamps=True, initial_prompt="简体")

    index = 1
    subs = []
    for segment in segments:
        subtitle = None
        for word in segment.words:
            if subtitle is None:
                subtitle = srt.Subtitle(index, datetime.timedelta(seconds=word.start), datetime.timedelta(seconds=word.end), "")
            final_word = word.word.strip()
            subtitle.end = datetime.timedelta(seconds=word.end)

            is_end = (
                final_word[-1] in end_interpunction
                and not (final_word[-1] == "." and len(final_word) > 1 and final_word[-2] in en_num_chars)
            )
            is_too_long = subtitle is not None and len(subtitle.content) > 20

            if is_end or is_too_long:
                push_word = final_word[:-1] if (is_end and not is_too_long) else final_word
                subtitle.content += push_word
                subs.append(subtitle)
                index += 1
                subtitle = None
            else:
                subtitle.content += final_word

        if subtitle is not None:
            subs.append(subtitle)
            index += 1

    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))


def srt_sentence_merge(source_path: str, output_path: str) -> None:
    """将词级字幕合并为句级字幕"""
    with open(source_path, "r", encoding="utf-8") as f:
        sub_list = list(srt.parse(f.read()))

    if not sub_list:
        logging.info("未找到字幕")
        return

    logging.info("开始字幕语句合并")
    merged = []
    current = None
    merge_index = 1

    for i, item in enumerate(sub_list):
        dot_idx = item.content.rfind(".")
        excl_idx = item.content.rfind("!")
        ques_idx = item.content.rfind("?")
        end_idx = max(dot_idx, excl_idx, ques_idx)

        if current is None:
            current = copy.copy(item)
            current.content = ""

        current.index = merge_index
        current.end = item.end
        current.content += item.content

        is_last = (i == len(sub_list) - 1)
        is_sentence_end = (end_idx == len(item.content) - 1)

        if is_last or is_sentence_end:
            merged.append(current)
            current = None
            merge_index += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(merged))
