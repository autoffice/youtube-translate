"""使用 ChatTTS 进行中文语音合成"""
import copy
import logging
import os
import re
from typing import Optional

import numpy as np
import srt
import torch
from pydub import AudioSegment


def _number_to_zh(num_str: str) -> str:
    """将阿拉伯数字转为中文口语数字"""
    digits = "零一二三四五六七八九"
    units = ["", "十", "百", "千"]
    num = int(num_str)

    if num == 0:
        return digits[0]
    if num < 10:
        return digits[num]
    if num < 100:
        tens, ones = divmod(num, 10)
        if tens == 1:
            return "十" + (digits[ones] if ones else "")
        return digits[tens] + "十" + (digits[ones] if ones else "")
    if num < 10000:
        parts = []
        num_chars = list(map(int, str(num)))
        length = len(num_chars)
        for i, n in enumerate(num_chars):
            pos = length - i - 1
            if n == 0:
                if parts and parts[-1] != digits[0] and any(x != 0 for x in num_chars[i + 1:]):
                    parts.append(digits[0])
                continue
            parts.append(digits[n] + units[pos])
        return "".join(parts).rstrip(digits[0])
    return "".join(digits[int(ch)] for ch in num_str)


def _normalize_tts_text(text: str) -> str:
    """在送入 ChatTTS 前做文本规范化，避免被其 normalizer 丢字"""
    text = text.strip()
    text = text.replace("／", "，")
    text = text.replace("/", "，")
    text = text.replace("：", "，")
    text = text.replace(":", "，")
    text = text.replace("；", "，")
    text = text.replace(";", "，")
    text = text.replace("！", "。")
    text = text.replace("!", "。")
    text = text.replace("？", "。")
    text = text.replace("?", "。")
    text = re.sub(r"\d+", lambda m: _number_to_zh(m.group(0)), text)
    text = re.sub(r"\s+", " ", text)
    return text


def srt_to_voice_chattts(
    srt_path: str,
    output_dir: str,
    seed: Optional[int] = None,
) -> bool:
    """使用 ChatTTS 将 SRT 字幕转为语音"""
    try:
        import ChatTTS
    except ImportError:
        logging.error("ChatTTS 未安装，请运行: pip install ChatTTS")
        return False

    os.makedirs(output_dir, exist_ok=True)

    logging.info("正在加载 ChatTTS 模型...")
    chat = ChatTTS.Chat()
    chat.load(source="huggingface", compile=False)
    logging.info("ChatTTS 模型加载完成")

    with open(srt_path, "r", encoding="utf-8") as f:
        sub_list = list(srt.parse(f.read()))

    if not sub_list:
        logging.warning("SRT 文件为空")
        return False

    if seed is not None:
        torch.manual_seed(seed)
        logging.info("使用音色种子: %d", seed)

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=chat.sample_random_speaker(),
        temperature=0.3,
        top_P=0.7,
        top_K=20,
    )
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt="[oral_2][laugh_0][break_4]",
    )

    file_names = []
    logging.info("开始生成语音，共 %d 条字幕", len(sub_list))

    for i, sub in enumerate(sub_list, 1):
        normalized_text = _normalize_tts_text(sub.content)
        if normalized_text != sub.content:
            logging.info("第 %d 条字幕已规范化: %s -> %s", i, sub.content, normalized_text)

        wavs = chat.infer(
            [normalized_text],
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
        )

        if not wavs or len(wavs) == 0:
            logging.warning("第 %d 条字幕生成失败，跳过", i)
            continue

        wav_data = wavs[0]
        if isinstance(wav_data, torch.Tensor):
            wav_data = wav_data.cpu().numpy()
        wav_data = wav_data.squeeze()
        # float32 -> int16 PCM
        wav_data = np.clip(wav_data, -1.0, 1.0)
        wav_int16 = (wav_data * 32767).astype(np.int16)

        file_name = f"{i}.wav"
        file_path = os.path.join(output_dir, file_name)
        file_names.append(file_name)

        audio = AudioSegment(
            wav_int16.tobytes(),
            frame_rate=24000,
            sample_width=2,
            channels=1,
        )
        audio.export(file_path, format="wav")

        if (i % 10 == 0) or (i == len(sub_list)):
            logging.info("已生成 %d/%d 条语音", i, len(sub_list))

    _save_voice_map(sub_list, file_names, output_dir, srt_path)
    logging.info("ChatTTS 语音生成完成")
    return True


def _save_voice_map(sub_list, file_names, output_dir, srt_path):
    """保存语音映射文件"""
    voice_map = copy.deepcopy(list(sub_list))
    for i, name in enumerate(file_names):
        voice_map[i].content = name

    with open(os.path.join(output_dir, "voiceMap.srt"), "w", encoding="utf-8") as f:
        f.write(srt.compose(voice_map))

    with open(srt_path, "r", encoding="utf-8") as src:
        with open(os.path.join(output_dir, "sub.srt"), "w", encoding="utf-8") as dst:
            dst.write(src.read())
