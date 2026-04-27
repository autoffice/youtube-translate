"""使用 ChatTTS 进行中文语音合成"""
import copy
import logging
import os
from typing import Optional

import srt
import torch
from pydub import AudioSegment


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

    texts = [sub.content for sub in sub_list]
    wavs = chat.infer(
        texts,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
    )

    for i, wav in enumerate(wavs, 1):
        file_name = f"{i}.wav"
        file_path = os.path.join(output_dir, file_name)
        file_names.append(file_name)

        audio = AudioSegment(
            wav.tobytes(),
            frame_rate=24000,
            sample_width=2,
            channels=1,
        )
        audio.export(file_path, format="wav")

        if (i % 10 == 0) or (i == len(wavs)):
            logging.info("已生成 %d/%d 条语音", i, len(wavs))

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
