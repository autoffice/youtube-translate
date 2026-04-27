"""
交互式重新生成特定字幕的语音
使用 ChatTTS 本地生成
"""
import json
import logging
import os
from typing import Optional

import srt
import torch
from pydub import AudioSegment


def _generate_single_voice(
    chat,
    text: str,
    output_file: str,
    seed: Optional[int] = None,
) -> bool:
    """使用 ChatTTS 生成单条语音"""
    import ChatTTS

    if seed is not None:
        torch.manual_seed(seed)

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=chat.sample_random_speaker(),
        temperature=0.3,
        top_P=0.7,
        top_K=20,
    )
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt="[oral_2][laugh_0][break_4]",
    )

    wavs = chat.infer(
        [text],
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
    )

    if not wavs or len(wavs) == 0:
        return False

    audio = AudioSegment(
        wavs[0].tobytes(),
        frame_rate=24000,
        sample_width=2,
        channels=1,
    )
    audio.export(output_file, format="wav")
    return True


def load_config(path: str) -> dict:
    """加载参数文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """交互式重新生成字幕语音"""
    try:
        import ChatTTS
    except ImportError:
        logging.error("ChatTTS 未安装，请运行: pip install ChatTTS")
        return

    param_path = input("请输入参数文件路径: ")
    if not os.path.exists(param_path):
        logging.error("参数文件不存在: %s", param_path)
        return

    param = load_config(param_path)
    video_id = param["视频ID"]
    work_path = param["工作目录"]
    voice_dir = os.path.join(work_path, f"{video_id}_zh_source")
    seed = param.get("TTS音色种子", 0)
    seed = seed if seed > 0 else None

    voice_map_path = os.path.join(voice_dir, "voiceMap.srt")
    srt_path = os.path.join(voice_dir, "sub.srt")

    with open(voice_map_path, "r", encoding="utf-8") as f:
        voice_map = list(srt.parse(f.read()))
    with open(srt_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    # 加载 ChatTTS 模型
    logging.info("正在加载 ChatTTS 模型...")
    chat = ChatTTS.Chat()
    chat.load(compile=False)
    logging.info("ChatTTS 模型加载完成")

    while True:
        try:
            index = int(input("请输入要重新合成的字幕序号（0 退出）: "))
            if index == 0:
                break
            if index < 1 or index > len(subtitles):
                logging.warning("序号超出范围")
                continue

            idx = index - 1
            logging.info("文件名: %s", voice_map[idx].content)
            logging.info("字幕内容: %s", subtitles[idx].content)

            voice_file = os.path.join(voice_dir, voice_map[idx].content)
            if os.path.exists(voice_file):
                os.remove(voice_file)
                logging.info("已删除旧文件: %s", voice_file)

            if _generate_single_voice(chat, subtitles[idx].content, voice_file, seed):
                logging.info("重新生成成功")
            else:
                logging.error("重新生成失败")
        except ValueError:
            logging.warning("请输入有效的数字")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
