"""视频下载与合成模块"""
import logging
import os
import subprocess
from typing import Dict, Optional


def _norm_path(p: Optional[str]) -> Optional[str]:
    """规范化路径为正斜杠（FFmpeg 兼容）"""
    return p.replace("\\", "/") if p else p


def download_video(
    video_id: str,
    output_path: str,
    proxies: Optional[Dict] = None,
    best_quality: bool = False,
) -> None:
    """使用 yt-dlp 下载 YouTube 视频"""
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
    }

    if best_quality:
        ydl_opts["format"] = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
        ydl_opts["merge_output_format"] = "mp4"
    else:
        ydl_opts["format"] = "best[ext=mp4][height<=480]/best[ext=mp4]/best"
        ydl_opts["merge_output_format"] = "mp4"

    if proxies:
        proxy_url = proxies.get("http") or proxies.get("https")
        if proxy_url:
            ydl_opts["proxy"] = proxy_url

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def compose_video(
    video_path: str,
    voice_path: Optional[str],
    instrument_path: Optional[str],
    srt_path: Optional[str],
    output_path: str,
) -> bool:
    """使用 FFmpeg 合成最终视频"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    video_path = _norm_path(video_path)
    voice_path = _norm_path(voice_path) if (voice_path and os.path.exists(voice_path)) else None
    inst_path = _norm_path(instrument_path) if (instrument_path and os.path.exists(instrument_path)) else None
    srt_path = _norm_path(srt_path) if (srt_path and os.path.exists(srt_path)) else None

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video_path]
    input_index = 1
    voice_idx = None
    inst_idx = None

    if voice_path:
        cmd += ["-i", voice_path]
        voice_idx = input_index
        input_index += 1
    if inst_path:
        cmd += ["-i", inst_path]
        inst_idx = input_index
        input_index += 1

    filter_complex = None
    maps = []

    has_both = (voice_idx is not None and inst_idx is not None)
    has_single = (voice_idx is not None) ^ (inst_idx is not None)

    if has_both:
        if srt_path:
            filter_complex = f"[{voice_idx}:a][{inst_idx}:a]amix=inputs=2[a];[0:v]subtitles={srt_path}[v]"
            maps = ["-map", "[v]", "-map", "[a]"]
        else:
            filter_complex = f"[{voice_idx}:a][{inst_idx}:a]amix=inputs=2[a]"
            maps = ["-map", "0:v", "-map", "[a]"]
    elif has_single:
        audio_idx = voice_idx if voice_idx is not None else inst_idx
        if srt_path:
            cmd += ["-vf", f"subtitles={srt_path}"]
            maps = ["-map", "0:v", "-map", f"{audio_idx}:a"]
        else:
            maps = ["-map", "0:v", "-map", f"{audio_idx}:a"]
    else:
        if srt_path:
            cmd += ["-vf", f"subtitles={srt_path}"]

    if filter_complex:
        cmd += ["-filter_complex", filter_complex]

    cmd += maps + ["-c:v", "libx264", "-c:a", "aac", "-shortest", output_path]

    logging.info("使用 FFmpeg 合成最终视频")
    subprocess.run(cmd, check=True)
    return True


def check_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("未安装 ffmpeg，请安装并添加到 PATH")
        return False
