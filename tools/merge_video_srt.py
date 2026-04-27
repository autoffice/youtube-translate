"""
使用 FFmpeg 合成视频、字幕和音频
"""
import logging
import subprocess
from typing import Optional


def merge_video_with_subtitles_and_audio(
    input_video: str,
    input_subtitle: str,
    input_audio1: str,
    input_audio2: str,
    output_video: str,
) -> None:
    """
    使用 FFmpeg 将字幕和两个音频文件合成到视频中

    Args:
        input_video: 输入视频文件路径
        input_subtitle: 字幕文件路径（ASS 格式）
        input_audio1: 第一个音频文件路径（例如乐器音轨）
        input_audio2: 第二个音频文件路径（例如人声音轨）
        output_video: 输出视频文件路径
    """
    logging.info("正在添加字幕和音频到视频中...")

    # 规范化路径为正斜杠（Windows 兼容）
    subtitle_path = input_subtitle.replace("\\", "/")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", input_video,
        "-i", input_audio1,
        "-i", input_audio2,
        "-filter_complex",
        f"[1:a][2:a]amix=inputs=2[a];[0:v]subtitles={subtitle_path}[v]",
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        output_video,
    ]

    logging.info("FFmpeg 命令: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logging.info("视频合成完成: %s", output_video)
