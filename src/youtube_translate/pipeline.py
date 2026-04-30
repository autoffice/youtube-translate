"""YouTube 视频自动翻译配音主流程"""
import argparse
import atexit
import io
import json
import logging
import os
import sys
import time
from typing import Optional

import srt
from pydub import AudioSegment

from youtube_translate.config import load_env_config
from youtube_translate.separator import audio_remove
from youtube_translate.transcriber import transcribe_audio_en, transcribe_audio_zh, srt_sentence_merge
from youtube_translate.translator import DashScopeTranslator
from youtube_translate.subtitle import merge_subtitles
from youtube_translate.tts import srt_to_voice_chattts
from youtube_translate.video import download_video, compose_video, check_ffmpeg
from youtube_translate.uploader import upload_to_bilibili

os.environ["PYTHONIOENCODING"] = "utf-8"

TERMS_FILE = os.path.join(os.path.dirname(__file__), "resources", "terms.json")


def _setup_logging() -> None:
    stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdout, "buffer") else sys.stdout
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=stream,
    )


def _srt_translate_dashscope(source_path: str, output_path: str, model_name: str, metadata_path: str = "") -> None:
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

    # 生成视频元数据（标题、标签、描述）
    if metadata_path:
        if os.path.exists(metadata_path):
            logging.info("视频元数据已存在，跳过生成: %s", metadata_path)
        else:
            try:
                metadata = translator.generate_video_metadata(translated)
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                logging.info("视频元数据已保存: %s", metadata_path)
            except Exception:
                logging.exception("视频元数据生成失败")


def _get_font_path() -> str:
    """根据操作系统选择最佳中文字体"""
    import platform
    system = platform.system()
    if system == "Windows":
        # 微软雅黑粗体
        candidates = [
            "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑 Bold
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
        ]
    else:
        # macOS 苹方粗体
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",                          # 苹方
            "/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",                    # 华文黑体
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            return path.replace("\\", "/")

    return "sans-serif"


def _generate_cover(video_path: str, cover_path: str, metadata_path: str) -> None:
    """从视频截取一帧并叠加标题文字生成封面"""
    import subprocess

    title = ""
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        title = metadata.get("title", "")

    # 从视频截取一帧作为底图
    frame_path = cover_path.replace(".jpg", "_frame.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path, "-vf", "select=eq(pict_type\\,I)", "-frames:v", "1",
        "-q:v", "2", frame_path,
    ]
    subprocess.run(cmd, check=True)

    if not title:
        os.rename(frame_path, cover_path)
        logging.info("封面已生成（无标题）: %s", cover_path)
        return

    font_path = _get_font_path().replace(":", "\\:")
    escaped_title = title.replace("'", "'\\''").replace(":", "\\:")

    # 黄色粗体 + 黑色阴影描边
    drawtext = (
        f"drawtext=text='{escaped_title}'"
        f":fontfile='{font_path}'"
        f":fontsize=56"
        f":fontcolor=yellow"
        f":shadowcolor=black:shadowx=3:shadowy=3"
        f":borderw=2:bordercolor=black"
        f":x=(w-text_w)/2:y=(h-text_h)/2"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", frame_path, "-vf", drawtext, cover_path,
    ]
    subprocess.run(cmd, check=True)

    if os.path.exists(frame_path):
        os.remove(frame_path)
    logging.info("封面已生成: %s", cover_path)


def _voice_connect(source_dir: str, output_path: str) -> bool:
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

    import datetime
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
                    speed_up = max_speed_up
                if speed_up < min_speed_up:
                    speed_up = min_speed_up
                audio = audio.speedup(playback_speed=speed_up)

        audio = audio.fade_in(crossfade_ms).fade_out(crossfade_ms)
        combined = combined.overlay(audio, position=position)

    combined.export(output_path, format="wav")
    return True


def main():
    """主函数"""
    start_time = time.time()

    def _print_elapsed():
        logging.info("总耗时: %.2f 秒", time.time() - start_time)

    atexit.register(_print_elapsed)
    _setup_logging()

    if not check_ffmpeg():
        sys.exit(-1)

    parser = argparse.ArgumentParser(description="YouTube 视频自动翻译配音工具")
    parser.add_argument("-v", "--video-id", dest="video_id", help="视频 ID（覆盖 .env 中的 VIDEO_ID）")
    args = parser.parse_args()

    config = load_env_config()
    video_id = args.video_id if args.video_id else config["VIDEO_ID"]
    if not video_id:
        logging.error("未指定视频 ID，请在 .env 中设置 VIDEO_ID 或使用 -v 参数")
        sys.exit(-1)

    base_output_dir = config["OUTPUT_DIR"]
    work_path = os.path.join(base_output_dir, video_id)
    model_path = config["AUDIO_SEPARATION_MODEL"]

    proxies = None
    proxy_url = os.getenv("DOWNLOAD_PROXY")
    if proxy_url:
        proxies = {"http": proxy_url, "https": proxy_url}
        logging.info("检测到下载代理配置: %s", proxy_url)

    os.makedirs(work_path, exist_ok=True)
    config["VIDEO_ID"] = video_id
    config["OUTPUT_DIR"] = work_path
    logging.info("配置:\n%s", json.dumps(config, indent=2, ensure_ascii=False))

    # 下载视频
    video_file = os.path.join(work_path, "01_video.mp4")
    if os.path.exists(video_file):
        logging.info("视频已存在，跳过下载: %s", video_file)
    else:
        try:
            download_video(video_id, video_file, proxies)
            logging.info("视频下载完成: %s", video_file)
        except Exception:
            logging.exception("视频下载失败")
            sys.exit(-1)

    # 下载高清视频
    fhd_file = os.path.join(work_path, "01_video_fhd.mp4")
    if os.path.exists(fhd_file):
        logging.info("高清视频已存在，跳过下载: %s", fhd_file)
    else:
        try:
            download_video(video_id, fhd_file, proxies, best_quality=True)
            logging.info("高清视频下载完成: %s", fhd_file)
        except Exception:
            logging.exception("高清视频下载失败（非致命错误）")

    # 提取音频
    audio_file = os.path.join(work_path, "02_audio.wav")
    if os.path.exists(audio_file):
        logging.info("音频已存在，跳过提取: %s", audio_file)
    else:
        try:
            import subprocess
            cmd = ["ffmpeg", "-i", video_file, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_file]
            subprocess.run(cmd, check=True, capture_output=True)
            logging.info("音频提取完成: %s", audio_file)
        except Exception:
            logging.exception("音频提取失败")
            sys.exit(-1)

    # 音频分离
    voice_file = os.path.join(work_path, "03_voice.wav")
    instrument_file = os.path.join(work_path, "03_instrument.wav")
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

    # 语音转写
    srt_en_file = os.path.join(work_path, "04_en.srt")
    if os.path.exists(srt_en_file):
        logging.info("英文字幕已存在，跳过转写: %s", srt_en_file)
    else:
        try:
            transcribe_audio_en(voice_file, config["WHISPER_MODEL"], "en", srt_en_file)
            logging.info("语音转写完成: %s", srt_en_file)
        except Exception:
            logging.exception("语音转写失败")
            sys.exit(-1)

    # 字幕语句合并
    srt_en_merge_file = os.path.join(work_path, "05_en_merge.srt")
    if os.path.exists(srt_en_merge_file):
        logging.info("合并字幕已存在，跳过合并: %s", srt_en_merge_file)
    else:
        try:
            srt_sentence_merge(srt_en_file, srt_en_merge_file)
            logging.info("字幕合并完成: %s", srt_en_merge_file)
        except Exception:
            logging.exception("字幕合并失败")
            sys.exit(-1)

    # 字幕翻译
    srt_zh_file = os.path.join(work_path, "06_zh.srt")
    metadata_file = os.path.join(work_path, "06_metadata.json")
    if os.path.exists(srt_zh_file):
        logging.info("中文字幕已存在，跳过翻译: %s", srt_zh_file)
    else:
        try:
            _srt_translate_dashscope(srt_en_merge_file, srt_zh_file, config["TRANSLATE_MODEL"], metadata_file)
            logging.info("字幕翻译完成: %s", srt_zh_file)
        except Exception:
            logging.exception("字幕翻译失败")
            sys.exit(-1)

    # 中文配音（可选）
    voice_dir = os.path.join(work_path, "07_tts_source")
    voice_connected_file = os.path.join(work_path, "08_zh_voice.wav")
    srt_voice_file = os.path.join(work_path, "09_zh.srt")
    enable_dubbing = config.get("ENABLE_DUBBING", False)

    if enable_dubbing:
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

        if os.path.exists(voice_connected_file):
            logging.info("拼接语音已存在，跳过拼接: %s", voice_connected_file)
        else:
            try:
                if _voice_connect(voice_dir, voice_connected_file):
                    logging.info("语音拼接完成: %s", voice_connected_file)
                else:
                    logging.error("语音拼接失败")
                    sys.exit(-1)
            except Exception:
                logging.exception("语音拼接失败")
                sys.exit(-1)

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
        dual_srt_file = os.path.join(work_path, "10_dual.ass")
        if os.path.exists(dual_srt_file):
            logging.info("双语字幕已存在，跳过生成: %s", dual_srt_file)
        else:
            try:
                merge_subtitles(
                    srt_zh_file, srt_en_merge_file, dual_srt_file,
                    config.get("DUAL_ZH_FONT", "Arial"), config.get("DUAL_ZH_FONTSIZE", 10),
                    config.get("DUAL_EN_FONT", "Arial"), config.get("DUAL_EN_FONTSIZE", 6),
                )
                logging.info("双语字幕生成完成: %s", dual_srt_file)
            except Exception:
                logging.exception("双语字幕生成失败")
        final_srt = dual_srt_file
    elif enable_dubbing:
        final_srt = srt_voice_file
    else:
        final_srt = srt_zh_file

    # 合成最终视频
    output_file = os.path.join(work_path, "11_output.mp4")
    if os.path.exists(output_file):
        logging.info("最终视频已存在，跳过合成: %s", output_file)
    else:
        try:
            source_video = fhd_file if os.path.exists(fhd_file) else video_file
            if not os.path.exists(source_video):
                logging.error("找不到源视频文件")
                sys.exit(-1)

            if enable_dubbing:
                compose_video(source_video, voice_connected_file, instrument_file, final_srt, output_file)
            else:
                compose_video(source_video, None, None, final_srt, output_file)
            logging.info("最终视频生成完成: %s", output_file)
        except Exception:
            logging.exception("最终视频生成失败")
            sys.exit(-1)

    logging.info("全部完成！")
    logging.info("输出目录: %s", work_path)

    # 生成封面图片
    cover_file = os.path.join(work_path, "12_cover.jpg")
    if os.path.exists(cover_file):
        logging.info("封面已存在，跳过生成: %s", cover_file)
    else:
        try:
            _generate_cover(output_file, cover_file, metadata_file)
        except Exception:
            logging.exception("封面生成失败")

    # 上传到 B 站
    if config.get("BILIBILI_UPLOAD"):
        cookie_file = config.get("BILIBILI_COOKIE", "")
        if not cookie_file:
            logging.error("未配置 BILIBILI_COOKIE，跳过上传")
        elif not os.path.exists(metadata_file):
            logging.error("元数据文件不存在，跳过上传: %s", metadata_file)
        else:
            try:
                source_url = f"https://www.youtube.com/watch?v={video_id}" if config.get("BILIBILI_COPYRIGHT") == 2 else ""
                upload_to_bilibili(
                    video_path=output_file,
                    cover_path=cover_file if os.path.exists(cover_file) else "",
                    metadata_path=metadata_file,
                    cookie_file=cookie_file,
                    tid=config.get("BILIBILI_TID", 188),
                    copyright=config.get("BILIBILI_COPYRIGHT", 1),
                    source=source_url,
                    publish=config.get("BILIBILI_PUBLISH", False),
                )
            except Exception:
                logging.exception("B 站上传失败")
