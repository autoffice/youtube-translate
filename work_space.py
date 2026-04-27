"""
YouTube 视频自动翻译配音工具
支持平台：Windows (CUDA/CPU)、macOS (MPS/CPU)
"""
import argparse
import atexit
import copy
import datetime
import json
import logging
import math
import os
import struct
import subprocess
import sys
import time
import wave
from typing import Dict, List, Optional

import srt
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm

from tools.audio_remove import audio_remove
from tools.trans_dashscope import DashScopeTranslator
from tools.merge_subtitle import merge_subtitles
from tools.tts_chattts import srt_to_voice_chattts

# 设置编码（Windows 兼容）
os.environ["PYTHONIOENCODING"] = "utf-8"

# 加载环境变量
load_dotenv()

# 常量定义
TERMS_FILE = "tools/terms.json"


def _setup_logging() -> None:
    """配置日志系统"""
    import io
    stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdout, "buffer") else sys.stdout
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=stream,
    )


def _enable_whisper_debug() -> None:
    """启用 Whisper 调试日志"""
    os.environ["CT2_VERBOSE"] = "1"
    for name in ("faster_whisper", "ctranslate2"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True


def load_env_config() -> Dict:
    """从环境变量加载配置"""
    return {
        "VIDEO_ID": os.getenv("VIDEO_ID", ""),
        "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "./output"),
        "AUDIO_SEPARATION_MODEL": os.getenv("AUDIO_SEPARATION_MODEL", "models/audio_separation/baseline.pth"),
        "WHISPER_MODEL": os.getenv("WHISPER_MODEL", "medium"),
        "WHISPER_ZH_MODEL": os.getenv("WHISPER_ZH_MODEL", "medium"),
        "TRANSLATE_MODEL": os.getenv("TRANSLATE_MODEL", "qwen3.5-plus"),
        "DUAL_SUBTITLE": os.getenv("DUAL_SUBTITLE", "true").lower() == "true",
        "DUAL_ZH_FONT": os.getenv("DUAL_ZH_FONT", "Arial"),
        "DUAL_ZH_FONTSIZE": int(os.getenv("DUAL_ZH_FONTSIZE", "10")),
        "DUAL_EN_FONT": os.getenv("DUAL_EN_FONT", "Arial"),
        "DUAL_EN_FONTSIZE": int(os.getenv("DUAL_EN_FONTSIZE", "8")),
        "ENABLE_DUBBING": os.getenv("ENABLE_DUBBING", "false").lower() == "true",
        "TTS_SPEAKER_SEED": int(os.getenv("TTS_SPEAKER_SEED", "42")),
    }


def load_config(path: str) -> Dict:
    """保留兼容接口，实际从环境变量加载配置"""
    return load_env_config()


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
    英文语音转文字

    Args:
        audio_path: 音频文件路径
        model_name: Whisper 模型名称
        language: 语言代码
        output_srt_path: 输出 SRT 文件路径

    Returns:
        是否成功
    """
    not_silence_threshold_db = -30
    end_interpunction = ["…", ".", "!", "?", ";"]
    number_characters = "0123456789"

    initial_prompt = "简体" if language == "zh" else None
    device, compute_type = _get_whisper_device()

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root="models/whisper",
        local_files_only=True,
    )
    logging.info("Whisper 模型已加载")

    segments, info = model.transcribe(
        audio=audio_path,
        language=language,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        log_progress=True,
    )

    # 转换为 SRT 字幕
    index = 1
    subs = []
    subtitle = None

    segments_list = list(segments)

    for segment in segments_list:
        for word in segment.words:
            if subtitle is None:
                subtitle = srt.Subtitle(
                    index,
                    datetime.timedelta(seconds=word.start),
                    datetime.timedelta(seconds=word.end),
                    "",
                )

            final_word = word.word.strip()
            subtitle.end = datetime.timedelta(seconds=word.end)

            # 判断是否句子结束
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
            max_volume = max(sample_volumes)

            if max_volume > not_silence_threshold:
                new_start_time = start_time + i / frame_rate
                break

        sub.start = datetime.timedelta(seconds=new_start_time)

    content = srt.compose(subs)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(content)

    return True


def transcribe_audio_zh(
    audio_path: str,
    model_name: str,
    output_srt_path: str,
) -> None:
    """中文语音转文字"""
    end_interpunction = ["。", "！", "？", "…", "；", "，", "、", ",", ".", "!", "?", ";"]
    en_num_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    device, compute_type = _get_whisper_device()
    model = WhisperModel(
        model_name, device=device, compute_type=compute_type,
        download_root="models/whisper", local_files_only=False,
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


def srt_translate_dashscope(
    source_path: str,
    output_path: str,
    model_name: str = "qwen-plus",
) -> None:
    """使用 DashScope 翻译 SRT 字幕"""
    with open(source_path, "r", encoding="utf-8") as f:
        sub_list = list(srt.parse(f.read()))

    translator = DashScopeTranslator(model_name=model_name)
    if os.path.exists(TERMS_FILE):
        translator.load_terms(TERMS_FILE)

    texts = [sub.content for sub in sub_list]
    translated = translator.translate_batch(texts)

    for i, text in enumerate(translated):
        sub_list[i].content = text

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(sub_list))


def _norm_path(p: Optional[str]) -> Optional[str]:
    """规范化路径为正斜杠（FFmpeg 兼容）"""
    return p.replace("\\", "/") if p else p


def voice_connect(source_dir: str, output_path: str) -> bool:
    """拼接语音片段，自动调整速度以匹配时间轴"""
    max_speed_up = 1.4
    min_speed_up = 1.2
    min_gap_duration = 0.1
    crossfade_ms = 30

    voice_map_path = os.path.join(source_dir, "voiceMap.srt")
    if not os.path.exists(voice_map_path):
        return False

    with open(voice_map_path, "r", encoding="utf-8") as f:
        voice_map = list(srt.parse(f.read()))

    duration = voice_map[-1].end.total_seconds() * 1000
    final_audio_path = os.path.join(source_dir, voice_map[-1].content)
    final_audio_end = voice_map[-1].start.total_seconds() * 1000
    final_audio_end += AudioSegment.from_wav(final_audio_path).duration_seconds * 1000
    duration = max(duration, final_audio_end)

    logging.info("开始语音拼接")
    combined = AudioSegment.silent(duration=duration)

    for i, item in enumerate(voice_map):
        audio_path = os.path.join(source_dir, item.content)
        audio = AudioSegment.from_wav(audio_path).strip_silence(silence_thresh=-40, silence_len=100)
        position = item.start.total_seconds() * 1000

        if i != len(voice_map) - 1:
            audio_end = position + audio.duration_seconds * 1000 + min_gap_duration * 1000
            next_position = voice_map[i + 1].start.total_seconds() * 1000

            if next_position < audio_end:
                speed_up = (audio.duration_seconds * 1000 + min_gap_duration * 1000) / (next_position - position)
                if speed_up > max_speed_up:
                    logging.warning("音频 %d 需要加速 %.2f 倍，超过最大值 %.2f，将截断", i + 1, speed_up, max_speed_up)
                    speed_up = max_speed_up
                if speed_up < min_speed_up:
                    logging.warning("音频 %d 加速倍率 %.2f 过低，强制设为 %.2f", i + 1, speed_up, min_speed_up)
                    speed_up = min_speed_up
                audio = audio.speedup(playback_speed=speed_up)

        audio = audio.fade_in(crossfade_ms).fade_out(crossfade_ms)
        combined = combined.overlay(audio, position=position)

    combined.export(output_path, format="wav")
    return True


def zh_video_compose(
    video_path: str,
    voice_path: Optional[str],
    instrument_path: Optional[str],
    srt_path: Optional[str],
    output_path: str,
) -> bool:
    """使用 FFmpeg 合成预览视频"""
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

    logging.info("使用 FFmpeg 生成预览视频")
    subprocess.run(cmd, check=True)
    return True


def env_check() -> bool:
    """检查环境依赖"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("未安装 ffmpeg，请安装并添加到 PATH")
        return False


def _download_video(
    video_id: str,
    output_path: str,
    proxies: Optional[Dict] = None,
    best_quality: bool = False,
) -> None:
    """
    使用 yt-dlp 下载 YouTube 视频

    Args:
        video_id: YouTube 视频 ID
        output_path: 输出文件路径
        proxies: 代理配置
        best_quality: 是否下载最高画质
    """
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


def main():
    """主函数"""
    start_time = time.time()

    def _print_elapsed():
        logging.info("总耗时: %.2f 秒", time.time() - start_time)

    atexit.register(_print_elapsed)

    _setup_logging()
    # _enable_whisper_debug()  # 需要时取消注释

    if not env_check():
        sys.exit(-1)

    parser = argparse.ArgumentParser(description="YouTube 视频自动翻译配音工具")
    parser.add_argument("-v", "--video-id", dest="video_id", help="视频 ID（覆盖 .env 中的 VIDEO_ID）")
    args = parser.parse_args()

    config = load_env_config()
    video_id = args.video_id if args.video_id else config["VIDEO_ID"]
    if not video_id:
        logging.error("未指定视频 ID，请在 .env 中设置 VIDEO_ID 或使用 -v 参数")
        sys.exit(-1)

    # 输出目录为 output/{video_id}
    base_output_dir = config["OUTPUT_DIR"]
    work_path = os.path.join(base_output_dir, video_id)
    model_path = config["AUDIO_SEPARATION_MODEL"]

    proxies = None
    proxy_url = os.getenv("DOWNLOAD_PROXY")
    if proxy_url:
        proxies = {"http": proxy_url, "https": proxy_url}
        logging.info("检测到下载代理配置: %s", proxy_url)

    os.makedirs(work_path, exist_ok=True)
    # 用实际生效的值覆盖配置，再输出日志
    config["VIDEO_ID"] = video_id
    config["OUTPUT_DIR"] = work_path
    logging.info("配置:\n%s", json.dumps(config, indent=2, ensure_ascii=False))

    # 下载视频（文件不存在时自动下载）
    video_file = os.path.join(work_path, f"{video_id}.mp4")
    if os.path.exists(video_file):
        logging.info("视频已存在，跳过下载: %s", video_file)
    else:
        try:
            _download_video(video_id, video_file, proxies)
            logging.info("视频下载完成: %s", video_file)
        except Exception:
            logging.exception("视频下载失败")
            sys.exit(-1)

    # 下载高清视频（文件不存在时自动下载，失败不影响后续流程）
    fhd_file = os.path.join(work_path, f"{video_id}_fhd.mp4")
    if os.path.exists(fhd_file):
        logging.info("高清视频已存在，跳过下载: %s", fhd_file)
    else:
        try:
            _download_video(video_id, fhd_file, proxies, best_quality=True)
            logging.info("高清视频下载完成: %s", fhd_file)
        except Exception:
            logging.exception("高清视频下载失败（非致命错误）")

    # 提取音频（文件不存在时自动执行）
    audio_file = os.path.join(work_path, f"{video_id}.wav")
    if os.path.exists(audio_file):
        logging.info("音频已存在，跳过提取: %s", audio_file)
    else:
        try:
            video = VideoFileClip(video_file)
            video.audio.write_audiofile(audio_file, logger=None)
            logging.info("音频提取完成: %s", audio_file)
        except Exception:
            logging.exception("音频提取失败")
            sys.exit(-1)

    # 音频分离（文件不存在时自动执行）
    voice_file = os.path.join(work_path, f"{video_id}_voice.wav")
    instrument_file = os.path.join(work_path, f"{video_id}_instrument.wav")
    if os.path.exists(voice_file) and os.path.exists(instrument_file):
        logging.info("人声和伴奏已存在，跳过分离")
        logging.info("  人声: %s", voice_file)
        logging.info("  伴奏: %s", instrument_file)
    else:
        try:
            audio_remove(audio_file, voice_file, instrument_file, model_path)
            logging.info("音频分离完成")
        except Exception:
            logging.exception("音频分离失败")
            sys.exit(-1)

    # 语音转写（文件不存在时自动执行）
    srt_en_file = os.path.join(work_path, f"{video_id}_en.srt")
    if os.path.exists(srt_en_file):
        logging.info("英文字幕已存在，跳过转写: %s", srt_en_file)
    else:
        try:
            transcribe_audio_en(voice_file, config["WHISPER_MODEL"], "en", srt_en_file)
            logging.info("语音转写完成: %s", srt_en_file)
        except Exception:
            logging.exception("语音转写失败")
            sys.exit(-1)

    # 字幕语句合并（文件不存在时自动执行）
    srt_en_merge_file = os.path.join(work_path, f"{video_id}_en_merge.srt")
    if os.path.exists(srt_en_merge_file):
        logging.info("合并字幕已存在，跳过合并: %s", srt_en_merge_file)
    else:
        try:
            srt_sentence_merge(srt_en_file, srt_en_merge_file)
            logging.info("字幕合并完成: %s", srt_en_merge_file)
        except Exception:
            logging.exception("字幕合并失败")
            sys.exit(-1)

    # 字幕翻译（文件不存在时自动执行）
    srt_zh_file = os.path.join(work_path, f"{video_id}_zh_merge.srt")
    if os.path.exists(srt_zh_file):
        logging.info("中文字幕已存在，跳过翻译: %s", srt_zh_file)
    else:
        try:
            srt_translate_dashscope(srt_en_merge_file, srt_zh_file, config["TRANSLATE_MODEL"])
            logging.info("字幕翻译完成: %s", srt_zh_file)
        except Exception:
            logging.exception("字幕翻译失败")
            sys.exit(-1)

    # 字幕转语音（可选，由配置控制）
    voice_dir = os.path.join(work_path, f"{video_id}_zh_source")
    voice_connected_file = os.path.join(work_path, f"{video_id}_zh.wav")
    srt_voice_file = os.path.join(work_path, f"{video_id}_zh.srt")
    enable_dubbing = config.get("ENABLE_DUBBING", False)

    if enable_dubbing:
        # TTS 生成语音片段（检测 voiceMap.srt 判断是否已完成）
        voice_map_file = os.path.join(voice_dir, "voiceMap.srt")
        if os.path.exists(voice_map_file):
            logging.info("语音片段已存在，跳过 TTS: %s", voice_dir)
        else:
            try:
                seed = config.get("TTS_SPEAKER_SEED", 0)
                seed = seed if seed > 0 else None
                srt_to_voice_chattts(srt_zh_file, voice_dir, seed)
                logging.info("字幕转语音完成: %s", voice_dir)
            except Exception:
                logging.exception("字幕转语音失败")
                sys.exit(-1)

        # 语音拼接（文件不存在时自动执行）
        if os.path.exists(voice_connected_file):
            logging.info("拼接语音已存在，跳过拼接: %s", voice_connected_file)
        else:
            try:
                if voice_connect(voice_dir, voice_connected_file):
                    logging.info("语音拼接完成: %s", voice_connected_file)
                else:
                    logging.error("语音拼接失败")
                    sys.exit(-1)
            except Exception:
                logging.exception("语音拼接失败")
                sys.exit(-1)

        # 中文语音转写（文件不存在时自动执行）
        if os.path.exists(srt_voice_file):
            logging.info("中文字幕已存在，跳过转写: %s", srt_voice_file)
        else:
            try:
                transcribe_audio_zh(voice_connected_file, config.get("WHISPER_ZH_MODEL", "medium"), srt_voice_file)
                logging.info("中文语音转写完成: %s", srt_voice_file)
            except Exception:
                logging.exception("中文语音转写失败")
                sys.exit(-1)

    # 确定最终字幕文件
    if config.get("DUAL_SUBTITLE"):
        dual_srt_file = os.path.join(work_path, f"{video_id}_dual.ass")
        if os.path.exists(dual_srt_file):
            logging.info("双语字幕已存在，跳过生成: %s", dual_srt_file)
        else:
            try:
                merge_subtitles(
                    srt_zh_file,
                    srt_en_merge_file,
                    dual_srt_file,
                    config.get("DUAL_ZH_FONT", "Arial"),
                    config.get("DUAL_ZH_FONTSIZE", 10),
                    config.get("DUAL_EN_FONT", "Arial"),
                    config.get("DUAL_EN_FONTSIZE", 6),
                )
                logging.info("双语字幕生成完成: %s", dual_srt_file)
            except Exception:
                logging.exception("双语字幕生成失败")
        final_srt = dual_srt_file
    elif enable_dubbing:
        final_srt = srt_voice_file
    else:
        final_srt = srt_zh_file

    # 合成最终视频（文件不存在时自动执行）
    output_file = os.path.join(work_path, f"{video_id}_output.mp4")
    if os.path.exists(output_file):
        logging.info("最终视频已存在，跳过合成: %s", output_file)
    else:
        try:
            source_video = fhd_file if os.path.exists(fhd_file) else video_file
            if not os.path.exists(source_video):
                logging.error("找不到源视频文件")
                sys.exit(-1)

            if enable_dubbing:
                # 配音开启：使用中文配音 + 背景音乐
                zh_video_compose(source_video, voice_connected_file, instrument_file, final_srt, output_file)
            else:
                # 配音关闭：保留原始声音，只加字幕
                zh_video_compose(source_video, None, None, final_srt, output_file)
            logging.info("最终视频生成完成: %s", output_file)
        except Exception:
            logging.exception("最终视频生成失败")
            sys.exit(-1)

    logging.info("全部完成！")
    logging.info("输出目录: %s", work_path)


if __name__ == "__main__":
    main()

