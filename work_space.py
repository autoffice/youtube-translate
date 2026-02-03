from tools.audio_remove import audio_remove

import os
import copy
import json
from pytubefix import YouTube
from pytubefix.cli import on_progress
from faster_whisper import WhisperModel
import srt
from pygtrans import Translate
import requests
from tqdm import tqdm
from pydub import AudioSegment
import asyncio
import edge_tts
import datetime
from moviepy.editor import VideoFileClip
import sys
import deepl
import wave
import math
import struct
from tools.trans_llm import TranslatorClass
import tenacity
import subprocess
import torch
import logging
import argparse
import atexit
import time

TTS_MAX_TRY_TIMES = 16
CHATGPT_URL = "https://api.openai.com/v1/"
GHATGPT_TERMS_FILE = "tools/terms.json"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 默认utf-8编码
os.environ['PYTHONIOENCODING'] = 'utf-8'


def enable_whisper_debug():
    # 让 CTranslate2 输出更详细的日志
    os.environ["CT2_VERBOSE"] = "1"

    # 仅提升 faster_whisper 与 ctranslate2 的日志级别到 DEBUG，交由全局 handler 处理
    for name in ("faster_whisper", "ctranslate2"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True


def load_param(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def transcribe_audio_en(path, model_name="base.en", language="en", srt_file_path_and_name="VIDEO_FILENAME.srt"):
    # 非静音检测阈值，单位为分贝，越小越严格
    not_silence_threshold_db = -30

    end_interpunction = ["…", ".", "!", "?", ";"]
    number_characters = "0123456789"
    # 确保简体中文
    initial_prompt = None
    if language == "zh":
        initial_prompt = "简体"

    if torch.cuda.is_available():
        device = 'cuda'
        compute_type = 'float16'
    else:
        device = 'cpu'
        compute_type = 'int8'

    model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root="faster-whisper_models", local_files_only=False)
    logging.info("Whisper model loaded.")
    segments, _ = model.transcribe(audio=path, language=language, word_timestamps=True, initial_prompt=initial_prompt)

    # 转换为srt的Subtitle对象
    index = 1
    subs = []
    subtitle = None
    for segment in segments:
        for word in segment.words:
            if subtitle is None:
                subtitle = srt.Subtitle(index, datetime.timedelta(seconds=word.start), datetime.timedelta(seconds=word.end), "")
            final_word = word.word.strip()
            subtitle.end = datetime.timedelta(seconds=word.end)

            # 避免ascii编码错误，不知道怎么写，以后再说吧
            # bytes_s = bytes(final_word, 'latin-1')  # Convert the string to bytes using latin-1 encoding
            # final_word = bytes_s.decode('latin-1')  # Decode the bytes to a string using utf-8 encoding
            # final_word = final_word.encode('utf-8')

            # 一句结束。但是要特别排除小数点被误认为是一句结尾的情况。
            if (final_word[-1] in end_interpunction) and not (len(final_word) > 1 and final_word[-2] in number_characters):
                push_word = " " + final_word
                subtitle.content += push_word
                subs.append(subtitle)
                index += 1
                subtitle = None

            else:
                if subtitle.content == "":
                    subtitle.content = final_word
                # 如果上一个字符是"."，则要考虑小数的可能性
                elif final_word[0] == ".":
                    subtitle.content = subtitle.content + final_word
                elif len(subtitle.content) > 0 and subtitle.content[-1] == "." and final_word[0] in number_characters:
                    subtitle.content = subtitle.content + final_word
                else:
                    subtitle.content = subtitle.content + " " + final_word
    # 补充最后一个字幕 
    if subtitle is not None:
        subs.append(subtitle)
        index += 1

    logging.info("Transcription complete.")

    # 重新校准字幕开头，以字幕开始时间后声音大于阈值的第一帧为准
    audio_wav = wave.open(path, 'rb')
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
            sample_volumes = []  # 用于存储每个样本的音量值
            for sample_tuple in samples:
                # sample是一个样本值
                # 调用calculate_volume函数计算样本的音量值，并将结果添加到sampleVolumes列表中
                sample = sample_tuple[0]
                sample_volume = abs(sample) / 32768
                sample_volumes.append(sample_volume)  # 将音量值添加到列表中
            # 找出所有样本的音量值中的最大值
            max_volume = max(sample_volumes)

            if max_volume > not_silence_threshold:
                new_start_time = start_time + i / frame_rate
                break

        sub.start = datetime.timedelta(seconds=new_start_time)

    content = srt.compose(subs)
    with open(srt_file_path_and_name, "w", encoding="utf-8") as file:
        file.write(content)

    logging.info("SRT file created.")
    logging.info("Output file: " + srt_file_path_and_name)
    return True


def transcribe_audio_zh(path, model_name="base.en", srt_file_path_and_name="VIDEO_FILENAME.srt"):
    end_interpunction = ["。", "！", "？", "…", "；", "，", "、", ",", ".", "!", "?", ";"]
    english_and_number_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    if torch.cuda.is_available():
        device = 'cuda'
        compute_type = 'float16'
    else:
        device = 'cpu'
        compute_type = 'int8'

    model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root="faster-whisper_models", local_files_only=False)
    segments, _ = model.transcribe(audio=path, language="zh", word_timestamps=True, initial_prompt="简体")

    index = 1
    subs = []
    for segment in segments:
        subtitle = None
        for word in segment.words:
            if subtitle is None:
                subtitle = srt.Subtitle(index, datetime.timedelta(seconds=word.start), datetime.timedelta(seconds=word.end), "")
            final_word = word.word.strip()
            subtitle.end = datetime.timedelta(seconds=word.end)

            # 排除英文字母+. 情况
            if (final_word[-1] in end_interpunction and not (final_word[-1] == "." and len(final_word) > 1 and final_word[-2] in english_and_number_characters)) \
                    or (subtitle is not None and len(subtitle.content) > 20):
                if not ((final_word[-1] == "." and len(final_word) > 1 and final_word[-2] in english_and_number_characters) or (subtitle is not None and len(subtitle.content) > 20)):
                    push_word = final_word[:-1]
                else:
                    push_word = final_word
                subtitle.content += push_word
                subs.append(subtitle)
                index += 1
                subtitle = None
            else:
                subtitle.content += final_word

        if subtitle is not None:
            subs.append(subtitle)
            index += 1

    content = srt.compose(subs)
    with open(srt_file_path_and_name, "w", encoding="utf-8") as file:
        file.write(content)


def srt_sentance_merge(source_srt_file_path_and_name, output_srt_file_path_and_name):
    srt_content = open(source_srt_file_path_and_name, "r", encoding="utf-8").read()
    sub_generator = srt.parse(srt_content)
    sub_list = list(sub_generator)
    if len(sub_list) == 0:
        logging.info("No subtitle found.")
        return False

    logging.info("<Sentence Merge Section>")

    sub_porcessing_index = 1
    sub_item_list = []
    sub_item_processing = None
    for subItem in sub_list:
        dot_index = subItem.content.rfind('.')
        exclamation_index = subItem.content.rfind('!')
        question_index = subItem.content.rfind('?')
        end_sentence_index = max(dot_index, exclamation_index, question_index)

        # 异常情况，句号居然在中间
        if end_sentence_index != -1 and end_sentence_index != len(subItem.content) - 1:
            log_string = f"Warning: Sentence (index:{end_sentence_index}) not end at the end of the subtitle.\n"
            log_string += f"Content: {subItem.content}"
            logging.info(log_string)

        # 以后一个字幕，直接拼接送入就可以了
        if subItem == sub_list[-1]:
            if sub_item_processing is None:
                sub_item_processing = copy.copy(subItem)
                sub_item_list.append(sub_item_processing)
                break
            else:
                sub_item_processing.end = subItem.end
                sub_item_processing.content += subItem.content
                sub_item_list.append(sub_item_processing)
                break

        # 新处理一串字符，则拷贝
        if sub_item_processing is None:
            sub_item_processing = copy.copy(subItem)
            sub_item_processing.content = ''  # 清空内容是为了延续后面拼接的逻辑

        sub_item_processing.index = sub_porcessing_index
        sub_item_processing.end = subItem.end
        sub_item_processing.content += subItem.content
        # 如果一句话结束了，就把这一句话送入处理
        if end_sentence_index == len(subItem.content) - 1:
            sub_item_list.append(sub_item_processing)
            sub_item_processing = None
            sub_porcessing_index += 1

    srt_content = srt.compose(sub_item_list)
    # 如果打开错误则返回false
    with open(output_srt_file_path_and_name, "w", encoding="utf-8") as file:
        file.write(srt_content)


def google_trans(proxies, texts):
    if proxies['https'] == "":
        client = Translate()
    else:
        client = Translate(proxies={'https': proxies['https']})
    texts_response = client.translate(texts, target='zh')
    texts_translated = []
    for txtResponse in texts_response:
        texts_translated.append(txtResponse.translatedText)
    return texts_translated


def deepl_translate(texts, key):
    translator = deepl.Translator(key)
    # list to string
    text_en = ""
    for oneLine in texts:
        text_en += oneLine + "\n"

    text_zh = translator.translate_text(text_en, target_lang="zh")
    text_zh = str(text_zh)
    texts_zh = text_zh.split("\n")
    return texts_zh


def srt_file_google_tran(proxies, source_file_name_and_path, output_file_name_and_path):
    srt_content = open(source_file_name_and_path, "r", encoding="utf-8").read()
    sub_generator = srt.parse(srt_content)
    sub_title_list = list(sub_generator)
    content_list = []
    for subTitle in sub_title_list:
        content_list.append(subTitle.content)

    content_list = google_trans(proxies, content_list)

    for i in range(len(sub_title_list)):
        sub_title_list[i].content = content_list[i]

    srt_content = srt.compose(sub_title_list)
    with open(output_file_name_and_path, "w", encoding="utf-8") as file:
        file.write(srt_content)


def srt_file_deepl_tran(source_file_name_and_path, output_file_name_and_path, key):
    srt_content = open(source_file_name_and_path, "r", encoding="utf-8").read()
    sub_generator = srt.parse(srt_content)
    sub_title_list = list(sub_generator)
    content_list = []
    for subTitle in sub_title_list:
        content_list.append(subTitle.content)

    content_list = deepl_translate(content_list, key)

    for i in range(len(sub_title_list)):
        sub_title_list[i].content = content_list[i]

    srt_content = srt.compose(sub_title_list)
    with open(output_file_name_and_path, "w", encoding="utf-8") as file:
        file.write(srt_content)


def gpt_translate(texts, key, model, proxies):
    translator = TranslatorClass(api_key=key,
                                 base_url=CHATGPT_URL,
                                 model_name=model,
                                 proxies=proxies)
    # 加载术语文件
    translator.load_terms(GHATGPT_TERMS_FILE)
    # list to string
    text_en = ""
    for oneLine in texts:
        text_en += oneLine + "\n"
    batch_text = text_en.split("\n")
    logging.info("Start to translate by GPT with Batch mode.")
    results = translator.translate_batch(batch_text, max_tokens=1200)
    texts_zh = []
    for i, result in enumerate(results, 1):
        logging.info(f"Translated text {i}: {result['text_result']}")
        logging.info(f"Process time {i}: {result['time']}")
        texts_zh.append(result['text_result'])
    return texts_zh


def srt_file_gpt_tran(model, proxies, source_file_name_and_path, output_file_name_and_path, key):
    srt_content = open(source_file_name_and_path, "r", encoding="utf-8").read()
    sub_generator = srt.parse(srt_content)
    sub_title_list = list(sub_generator)
    content_list = []
    for subTitle in sub_title_list:
        content_list.append(subTitle.content)

    content_list = gpt_translate(content_list, key, model, proxies)

    for i in range(len(sub_title_list)):
        sub_title_list[i].content = content_list[i]

    srt_content = srt.compose(sub_title_list)
    with open(output_file_name_and_path, "w", encoding="utf-8") as file:
        file.write(srt_content)


def string_to_voice(url, string, output_file):
    data = {
        "text": string,
        "text_language": "zh"
    }
    response = requests.post(url, json=data)
    if response.status_code != 200:
        return False

    with open(output_file, "wb") as f:
        f.write(response.content)

    return True


def srt_to_voice(url, srt_file_name_and_path, output_dir):
    # create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    srt_content = open(srt_file_name_and_path, "r", encoding="utf-8").read()
    sub_generator = srt.parse(srt_content)
    sub_title_list = list(sub_generator)
    index = 1
    file_names = []
    logging.info("Start to convert srt to voice")
    with tqdm(total=len(sub_title_list)) as pbar:
        for subTitle in sub_title_list:
            string = subTitle.content
            file_name = str(index) + ".wav"
            output_name_and_path = os.path.join(output_dir, file_name)
            file_names.append(file_name)
            try_times = 0

            while try_times < TTS_MAX_TRY_TIMES:
                if not string_to_voice(url, string, output_name_and_path):
                    return False

                # 获取outputNameAndPath的时间长度
                audio_wav = AudioSegment.from_wav(output_name_and_path)
                duration = len(audio_wav)
                # 获取最大音量
                max_volume = audio_wav.max_dBFS

                # 如果音频长度小于500ms，则重试，应该是数据有问题了
                if duration > 600 and max_volume > -15:
                    break

                try_times += 1

            if try_times >= TTS_MAX_TRY_TIMES:
                logging.info(f"Warning Failed to convert {file_name} to voice.")
                logging.info(f"Convert {file_name} duration: {duration}ms, max volume: {max_volume}dB")

            index += 1
            pbar.update(1)  # update progress bar

    voice_map_srt = copy.deepcopy(sub_title_list)
    for i in range(len(voice_map_srt)):
        voice_map_srt[i].content = file_names[i]
    voice_map_srt_content = srt.compose(voice_map_srt)
    voice_map_srt_file_and_path = os.path.join(output_dir, "voiceMap.srt")
    with open(voice_map_srt_file_and_path, "w", encoding="utf-8") as f:
        f.write(voice_map_srt_content)

    srt_atitional_file = os.path.join(output_dir, "zh.srt")
    with open(srt_atitional_file, "w", encoding="utf-8") as f:
        f.write(srt_content)

    logging.info("Convert srt to voice successfully")
    return True


@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                stop=tenacity.stop_after_attempt(5),
                reraise=True)
def srt_to_voice_edge(srt_file_name_and_path, output_dir, charactor="zh-CN-XiaoyiNeural"):
    # create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    srt_content = open(srt_file_name_and_path, "r", encoding="utf-8").read()
    sub_generator = srt.parse(srt_content)
    sub_title_list = list(sub_generator)
    index = 1
    file_names = []
    file_mp3_names = []

    async def convert_srt_to_voice_edge(text, path):
        logging.info(f"Start to convert srt to voice into {path}, text: {text}")
        communicate = edge_tts.Communicate(text, charactor)
        await communicate.save(path)

    coroutines = []
    for subTitle in sub_title_list:
        file_mp3_name = str(index) + ".mp3"
        file_name = str(index) + ".wav"
        output_mp3_name_and_path = os.path.join(output_dir, file_mp3_name)
        file_mp3_names.append(file_mp3_name)
        file_names.append(file_name)
        coroutines.append(convert_srt_to_voice_edge(subTitle.content, output_mp3_name_and_path))
        index += 1

    # wait for all coroutines to finish
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*coroutines))

    logging.info("Convert srt to mp3 voice successfully")

    # convert mp3 to wav
    for i in range(len(file_mp3_names)):
        mp3_file_name = file_mp3_names[i]
        wav_file_name = file_names[i]
        mp3_file_and_path = os.path.join(output_dir, mp3_file_name)
        wav_file_and_path = os.path.join(output_dir, wav_file_name)
        sound = AudioSegment.from_mp3(mp3_file_and_path)
        sound.export(wav_file_and_path, format="wav")
        os.remove(mp3_file_and_path)

    voice_map_srt = copy.deepcopy(sub_title_list)
    for i in range(len(voice_map_srt)):
        voice_map_srt[i].content = file_names[i]
    voice_map_srt_content = srt.compose(voice_map_srt)
    voice_map_srt_file_and_path = os.path.join(output_dir, "voiceMap.srt")
    with open(voice_map_srt_file_and_path, "w", encoding="utf-8") as f:
        f.write(voice_map_srt_content)

    srt_atitional_file = os.path.join(output_dir, "sub.srt")
    with open(srt_atitional_file, "w", encoding="utf-8") as f:
        f.write(srt_content)

    logging.info("Convert srt to wav voice successfully")
    return True


def zh_video_preview(video_file_name_and_path, voice_file_name_and_path, insturment_file_name_and_path, srt_file_name_and_path, output_file_name_and_path):
    """
    预览视频
    参数:
        video_file_name_and_path (str): 视频文件的路径和文件名
        voice_file_name_and_path (str): 音频文件的路径和文件名
        insturment_file_name_and_path (str): 乐器音频文件的路径和文件名
        srt_file_name_and_path (str): 字幕文件的路径和文件名
        output_file_name_and_path (str): 输出文件的路径和文件名
    返回:
        bool: 如果成功生成预览视频，则返回True，否则返回False
    """
    # 使用 FFmpeg 进行音频混音与字幕烧录
    if not os.path.exists(video_file_name_and_path):
        raise FileNotFoundError(f"Input video not found: {video_file_name_and_path}")

    # 规范化路径为正斜杠，避免 Windows 下 subtitles 过滤器路径转义问题
    def _norm(p):
        return p.replace("\\", "/") if p is not None else p

    video_path = _norm(video_file_name_and_path)
    voice_path = _norm(voice_file_name_and_path) if (voice_file_name_and_path and os.path.exists(voice_file_name_and_path)) else None
    inst_path = _norm(insturment_file_name_and_path) if (insturment_file_name_and_path and os.path.exists(insturment_file_name_and_path)) else None
    srt_path = _norm(srt_file_name_and_path) if (srt_file_name_and_path and os.path.exists(srt_file_name_and_path)) else None
    output_path = _norm(output_file_name_and_path)

    # 输入列表与索引跟踪
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', video_path
    ]
    input_index = 1  # 0 为视频
    voice_idx = None
    inst_idx = None

    if voice_path is not None:
        cmd += ['-i', voice_path]
        voice_idx = input_index
        input_index += 1
    if inst_path is not None:
        cmd += ['-i', inst_path]
        inst_idx = input_index
        input_index += 1

    filter_complex = None
    maps = []

    has_both_audios = (voice_idx is not None and inst_idx is not None)
    has_single_audio = (voice_idx is not None) ^ (inst_idx is not None)

    if has_both_audios:
        # 两路音频 -> amix；视频 -> 字幕（如有）
        if srt_path is not None:
            filter_complex = f"[{voice_idx}:a][{inst_idx}:a]amix=inputs=2[a];[0:v]subtitles={srt_path}[v]"
            maps = ['-map', '[v]', '-map', '[a]']
        else:
            filter_complex = f"[{voice_idx}:a][{inst_idx}:a]amix=inputs=2[a]"
            maps = ['-map', '0:v', '-map', '[a]']
    elif has_single_audio:
        # 单路音频：无需 amix；视频 -> 字幕（如有）
        audio_src_idx = voice_idx if voice_idx is not None else inst_idx
        if srt_path is not None:
            # 仅视频使用 filter；音频直接 map 输入流
            cmd += ['-vf', f'subtitles={srt_path}']
            maps = ['-map', '0:v', '-map', f'{audio_src_idx}:a']
        else:
            maps = ['-map', '0:v', '-map', f'{audio_src_idx}:a']
    else:
        # 无外部音频：仅字幕（如有），保留原视频音频（若存在）
        if srt_path is not None:
            cmd += ['-vf', f'subtitles={srt_path}']
        # 不显式 map，交由 ffmpeg 选择默认的音频流（若无则输出静音视频）

    if filter_complex is not None:
        cmd += ['-filter_complex', filter_complex]

    # 编解码与输出
    cmd += maps + ['-c:v', 'libx264', '-c:a', 'aac', '-shortest', output_path]

    logging.info("使用 FFmpeg 生成带字幕预览视频...")
    logging.info("Command: %s", ' '.join(cmd))
    subprocess.run(cmd, check=True)

    return True


def voice_connect(source_dir, output_and_path):
    max_speed_up = 1.4  # 最大音频加速（超过将截断，允许轻微重叠）
    min_speed_up = 1.2  # 最小音频加速
    min_gap_duration = 0.1  # 最小间隔时间，单位秒。低于这个间隔时间就认为音频重叠了
    crossfade_ms = 30  # 叠加处淡入淡出，降低重叠听感

    if not os.path.exists(source_dir):
        return False

    srt_map_file_name = "voiceMap.srt"
    srt_map_file_and_path = os.path.join(source_dir, srt_map_file_name)
    if not os.path.exists(srt_map_file_and_path):
        return False

    with open(srt_map_file_and_path, "r", encoding="utf-8") as f:
        voice_map_srt_content = f.read()

    # 确定音频长度
    voice_map_srt = list(srt.parse(voice_map_srt_content))
    duration = voice_map_srt[-1].end.total_seconds() * 1000
    final_audio_file_and_path = os.path.join(source_dir, voice_map_srt[-1].content)
    final_audio_end = voice_map_srt[-1].start.total_seconds() * 1000
    final_audio_end += AudioSegment.from_wav(final_audio_file_and_path).duration_seconds * 1000
    duration = max(duration, final_audio_end)

    logging.info("<Voice connect section>")

    # 初始化一个空的音频段
    combined = AudioSegment.silent(duration=duration)
    for i in range(len(voice_map_srt)):
        audio_file_and_path = os.path.join(source_dir, voice_map_srt[i].content)
        audio_wav = AudioSegment.from_wav(audio_file_and_path)
        audio_wav = audio_wav.strip_silence(silence_thresh=-40, silence_len=100)  # 去除头尾的静音
        audio_position = voice_map_srt[i].start.total_seconds() * 1000

        if i != len(voice_map_srt) - 1:
            # 检查上这一句的结尾到下一句的开头之间是否有静音，如果没有则需要缩小音频
            audio_end_position = audio_position + audio_wav.duration_seconds * 1000 + min_gap_duration * 1000
            audio_next_position = voice_map_srt[i + 1].start.total_seconds() * 1000
            if audio_next_position < audio_end_position:
                speed_up = (audio_wav.duration_seconds * 1000 + min_gap_duration * 1000) / (audio_next_position - audio_position)
                seconds = audio_position / 1000.0
                time_str = str(datetime.timedelta(seconds=seconds))
                if speed_up > max_speed_up:
                    # 超过可接受的最大变速，截断到最大值并允许轻微重叠
                    log_str = (
                        f"Warning: The audio_wav {i + 1} , at {time_str} , required speed up is {speed_up:.3f} > max_speed_up {max_speed_up}. "
                        f"Capping to {max_speed_up} and allowing slight overlap."
                    )
                    logging.info(log_str)
                    speed_up = max_speed_up

                # 音频如果提速一个略大于1，则speedup函数可能会出现一个错误的音频，所以这里确定最小的speedup为1.01
                if speed_up < min_speed_up:
                    log_str = f"Warning: The audio_wav {i + 1} , at {time_str} , speed up {speed_up} is too near to 1.0. Set to {min_speed_up} forcibly."
                    logging.info(log_str)
                    speed_up = min_speed_up
                audio_wav = audio_wav.speedup(playback_speed=speed_up)

        # 叠加前做轻微淡入淡出，降低边界处突兀感
        audio_wav = audio_wav.fade_in(crossfade_ms).fade_out(crossfade_ms)
        combined = combined.overlay(audio_wav, position=audio_position)

    combined.export(output_and_path, format="wav")
    return True


def env_check():
    # 检查环境变量中是否包含 ffmpeg
    # 尝试调用ffmpeg命令来检查其是否安装
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        ffmpeg_found = True
    except subprocess.CalledProcessError:
        # ffmpeg命令存在但执行出错（不太可能发生，除非ffmpeg损坏）
        ffmpeg_found = False
    except FileNotFoundError:
        # ffmpeg命令不存在
        ffmpeg_found = False

    waring_message = ""

    if not ffmpeg_found:
        waring_message += "未安装ffmpeg，请安装ffmpeg并将其所在目录添加到环境变量PATH中。\n"

    if waring_message:
        logging.info(f"环境依赖警告 {waring_message} ")
        return False
    else:
        return True


def main():
    # 统计运行耗时（秒），在正常或异常退出时打印
    _START_TS = time.time()

    def _print_elapsed():
        secs = time.time() - _START_TS
        logging.info(f"总耗时: {secs:.2f} 秒")

    atexit.register(_print_elapsed)

    # 全局基础日志：INFO 级别（统一 handler），带时间戳
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d %(funcName)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 打开 WhisperModel 的调试日志（仅设置等级，向上游传播）
    enable_whisper_debug()

    if not env_check():
        exit(-1)

    # 命令行参数：参数文件与 video_id
    parser = argparse.ArgumentParser(description="workspace runner")
    parser.add_argument('-p', '--param', dest='param_path', default='./example/1.json', help='参数文件路径，默认 ./example/1.json')
    parser.add_argument('-v', '--video-id', dest='video_id', default=None, help='视频 ID（可覆盖参数文件中的 video Id）')
    args = parser.parse_args()

    param_dict_path = args.param_path
    param_dict = load_param(param_dict_path)
    work_path = param_dict["work path"]
    video_id = args.video_id if args.video_id else param_dict["video Id"]
    audio_remove_model_name_and_path = param_dict["audio remove model path"]

    proxies = None if not param_dict["proxy"] else {
        'http': f"{param_dict["proxy"]}",
        'https': f"{param_dict["proxy"]}",
        'socks5': f"{param_dict["proxy"]}"
    }

    # create the working directory if it does not exist
    if not os.path.exists(work_path):
        os.makedirs(work_path)
        logging.info(f"Directory {work_path} created.")

    logging.info("配置\n" + json.dumps(param_dict, indent=4, ensure_ascii=False) + "\n")

    # 下载视频
    voice_file_name = f"{video_id}.mp4"
    viedo_file_name_and_path = os.path.join(work_path, voice_file_name)

    if param_dict["下载视频"]:
        logging.info(f"Downloading video {video_id} to {viedo_file_name_and_path}")
        try:
            # 如果已经有了，就不下载了
            if os.path.exists(viedo_file_name_and_path):
                logging.info(f"Video {video_id} already exists.")
                logging.info(f"[WORK -] Skip downloading video.")
            else:
                yt = YouTube(f'https://www.youtube.com/watch?v={video_id}', proxies=proxies, on_progress_callback=on_progress)
                video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').asc().first()
                video.download(output_path=work_path, filename=voice_file_name)
                # go back to the script directory
                logging.info(f"[WORK o] Download video {video_id} to {viedo_file_name_and_path} whith {video.resolution}.")
        except Exception:
            log_str = f"[WORK x] Error: Program blocked while downloading video {video_id} to {viedo_file_name_and_path}."
            logging.exception(log_str)
            sys.exit(-1)
    else:
        log_str = "[WORK -] Skip downloading video."
        logging.info(log_str)

    # try download more high-definition video
    # 需要单独下载最高分辨率视频，因为pytube下载的1080p视频没音频
    voice_fhd_file_name = f"{video_id}_fhd.mp4"
    voice_fhd_file_name_and_path = os.path.join(work_path, voice_fhd_file_name)
    if param_dict["下载高清视频"]:
        try:
            # 如果已经有了，就不下载了
            if os.path.exists(voice_fhd_file_name_and_path):
                logging.info(f"Video {video_id} already exists.")
                logging.info(f"[WORK -] Skip downloading video.")
            else:
                logging.info(f"Try to downloading more high-definition video {video_id} to {voice_fhd_file_name_and_path}")
                yt = YouTube(f'https://www.youtube.com/watch?v={video_id}', proxies=proxies, on_progress_callback=on_progress)
                video = yt.streams.filter(progressive=False, file_extension='mp4').order_by('resolution').desc().first()
                video.download(output_path=work_path, filename=voice_fhd_file_name)
                logging.info(f"[WORK o] Download 1080p high-definition {video_id} to {voice_fhd_file_name_and_path} whith {video.resolution}.")
        except Exception:
            log_str = f"[WORK x] Error: Program blocked while downloading high-definition video {video_id} to {voice_fhd_file_name_and_path}."
            logging.exception(log_str)
            log_str = f"Program will not exit for that the error is not critical."
            logging.info(log_str)
    else:
        log_str = "[WORK -] Skip downloading high-definition video."
        logging.info(log_str)

    # 视频转声音提取
    audio_file_name = f"{video_id}.wav"
    audio_file_name_and_path = os.path.join(work_path, audio_file_name)
    if param_dict["extract audio"]:
        # remove the audio file if it exists
        logging.info(f"Extracting audio from {viedo_file_name_and_path} to {audio_file_name_and_path}")
        try:
            video = VideoFileClip(viedo_file_name_and_path)
            audio = video.audio
            audio.write_audiofile(audio_file_name_and_path)
            logging.info(f"[WORK o] Extract audio from {viedo_file_name_and_path} to {audio_file_name_and_path} successfully.")
        except Exception:
            log_str = f"[WORK x] Error: Program blocked while extracting audio from {viedo_file_name_and_path} to {audio_file_name_and_path}."
            logging.exception(log_str)
            sys.exit(-1)
    else:
        log_str = "[WORK -] Skip extracting audio."
        logging.info(log_str)

    # 去除音频中的音乐
    voice_name = video_id + "_voice.wav"
    voice_name_and_path = os.path.join(work_path, voice_name)
    insturment_name = video_id + "_insturment.wav"
    insturment_name_and_path = os.path.join(work_path, insturment_name)
    if param_dict["audio remove"]:
        logging.info(f"Removing music from {audio_file_name_and_path} to {voice_name_and_path} and {insturment_name_and_path}")
        try:
            audio_remove(audio_file_name_and_path, voice_name_and_path, insturment_name_and_path, audio_remove_model_name_and_path)
            logging.info(f"[WORK o] Remove music from {audio_file_name_and_path} to {voice_name_and_path} and {insturment_name_and_path} successfully.")
        except Exception:
            log_str = f"[WORK x] Error: Program blocked while removing music from {audio_file_name_and_path} to {voice_name_and_path} and {insturment_name_and_path}."
            logging.exception(log_str)
            sys.exit(-1)
    else:
        log_str = "[WORK -] Skip removing music."
        logging.info(log_str)

    # 语音转文字
    srt_en_file_name = video_id + "_en.srt"
    srt_en_file_name_and_path = os.path.join(work_path, srt_en_file_name)
    if param_dict["audio transcribe"]:
        try:
            logging.info(f"Transcribing audio from {voice_name_and_path} to {srt_en_file_name_and_path}")
            transcribe_audio_en(voice_name_and_path, param_dict["audio transcribe model"], "en", srt_en_file_name_and_path)
            logging.info(f"[WORK o] Transcribe audio from {voice_name_and_path} to {srt_en_file_name_and_path} successfully.")
        except Exception:
            log_str = f"[WORK x] Error: Program blocked while transcribing audio from {voice_name_and_path} to {srt_en_file_name_and_path}."
            logging.exception(log_str)
            sys.exit(-1)
    else:
        log_str = "[WORK -] Skip transcription."
        logging.info(log_str)

    # 字幕语句合并
    srt_en_file_name_merge = video_id + "_en_merge.srt"
    srt_en_file_name_merge_and_path = os.path.join(work_path, srt_en_file_name_merge)
    if param_dict["srt merge"]:
        try:
            logging.info(f"Merging sentences in {srt_en_file_name_and_path} to {srt_en_file_name_merge_and_path}")
            srt_sentance_merge(srt_en_file_name_and_path, srt_en_file_name_merge_and_path)
            logging.info(f"[WORK o] Merge sentences in {srt_en_file_name_and_path} to {srt_en_file_name_merge_and_path} successfully.")
        except Exception:
            log_str = f"[WORK x] Error: Program blocked while merging sentences in {srt_en_file_name_and_path} to {srt_en_file_name_merge_and_path}."
            logging.exception(log_str)
            sys.exit(-1)
    else:
        log_str = "[WORK -] Skip sentence merge."
        logging.info(log_str)

    # 字幕翻译
    srt_zh_file_name = video_id + "_zh_merge.srt"
    srt_zh_file_name_and_path = os.path.join(work_path, srt_zh_file_name)
    if param_dict["srt merge translate"]:
        try:
            logging.info(f"Translating subtitle from {srt_en_file_name_merge_and_path} to {srt_zh_file_name_and_path}")
            if param_dict["srt merge translate tool"] == "deepl":
                if param_dict["srt merge translate key"] == "":
                    log_str = "[WORK x] Error: DeepL API key is not provided. Please provide it in the parameter file."
                    logging.info(log_str)
                    sys.exit(-1)
                srt_file_deepl_tran(srt_en_file_name_merge_and_path, srt_zh_file_name_and_path, param_dict["srt merge translate key"])
            elif 'gpt' in param_dict["srt merge translate tool"]:
                if param_dict['srt merge translate key'] == '':
                    log_str = "[WORK x] Error: GPT API key is not provided. Please provide it in the parameter file."
                    logging.info(log_str)
                    sys.exit(-1)
                srt_file_gpt_tran(param_dict['srt merge translate tool'],
                                  proxies,
                                  srt_en_file_name_merge_and_path,
                                  srt_zh_file_name_and_path,
                                  param_dict['srt merge translate key'])
            else:
                srt_file_google_tran(proxies, srt_en_file_name_merge_and_path, srt_zh_file_name_and_path)
                logging.info(f"[WORK o] Translate subtitle from {srt_en_file_name_merge_and_path} to {srt_zh_file_name_and_path} successfully.")
        except Exception:
            log_str = f"[WORK x] Error: Program blocked while translating subtitle from {srt_en_file_name_merge_and_path} to {srt_zh_file_name_and_path}."
            logging.exception(log_str)
            sys.exit(-1)
    else:
        log_str = "[WORK -] Skip subtitle translation."
        logging.info(log_str)

    # 字幕转语音
    tts_select = param_dict["TTS"]
    voice_dir = os.path.join(work_path, video_id + "_zh_source")
    if param_dict["srt to voice srouce"]:
        try:
            if tts_select == "GPT-SoVITS":
                logging.info(f"Converting subtitle to voice by GPT-SoVITS  in {srt_zh_file_name_and_path} to {voice_dir}")
                voice_url = param_dict["TTS param"]
                srt_to_voice(voice_url, srt_zh_file_name_and_path, voice_dir)
            else:
                charator = param_dict["TTS param"]
                if charator == "":
                    srt_to_voice_edge(srt_zh_file_name_and_path, voice_dir)
                else:
                    srt_to_voice_edge(srt_zh_file_name_and_path, voice_dir, charator)
                logging.info(f"Converting subtitle to voice by EdgeTTS in {srt_zh_file_name_and_path} to {voice_dir}")
            logging.info(f"[WORK o] Convert subtitle to voice in {srt_zh_file_name_and_path} to {voice_dir} successfully.")
        except Exception:
            log_str = f"[WORK x] Error: Program blocked while converting subtitle to voice in {srt_zh_file_name_and_path} to {voice_dir}."
            logging.exception(log_str)
            sys.exit(-1)
    else:
        log_str = "[WORK -] Skip voice conversion."
        logging.info(log_str)

    # 语音合并
    voice_connected_name = video_id + "_zh.wav"
    voice_connected_name_and_path = os.path.join(work_path, voice_connected_name)
    if param_dict["voice connect"]:
        try:
            logging.info(f"Connecting voice in {voice_dir} to {voice_connected_name_and_path}")
            ret = voice_connect(voice_dir, voice_connected_name_and_path)
            if ret:
                logging.info(f"[WORK o] Connect voice in {voice_dir} to {voice_connected_name_and_path} successfully.")
            else:
                logging.info(f"[WORK x] Connect voice in {voice_dir} to {voice_connected_name_and_path} failed.")
                sys.exit(-1)
        except Exception:
            log_str = f"[WORK x] Error: Program blocked while connecting voice in {voice_dir} to {voice_connected_name_and_path}."
            logging.exception(log_str)
            sys.exit(-1)
    else:
        log_str = "[WORK -] Skip voice connection."
        logging.info(log_str)

    # 合成后的语音转文字
    srt_voice_file_name = video_id + "_zh.srt"
    srt_voice_file_name_and_path = os.path.join(work_path, srt_voice_file_name)
    if param_dict["audio zh transcribe"]:
        try:
            if os.path.exists(srt_voice_file_name_and_path):
                logging.info("srt_voice_file_name_and_path exists.")
            else:
                logging.info(f"Transcribing audio from {voice_connected_name_and_path} to {srt_voice_file_name_and_path}")
                transcribe_audio_zh(voice_connected_name_and_path, param_dict["audio zh transcribe model"], srt_voice_file_name_and_path)
                logging.info(f"[WORK o] Transcribe audio from {voice_connected_name_and_path} to {srt_voice_file_name_and_path} successfully.")
        except Exception:
            log_str = f"[WORK x] Error: Program blocked while transcribing audio from {voice_connected_name_and_path} to {srt_voice_file_name_and_path}."
            logging.exception(log_str)
            sys.exit(-1)
    else:
        log_str = "[WORK -] Skip transcription."
        logging.info(log_str)

    # 合成预览视频
    preview_video_name = video_id + "_preview.mp4"
    preview_video_name_and_path = os.path.join(work_path, preview_video_name)
    if param_dict["video zh preview"]:
        try:
            if os.path.exists(voice_fhd_file_name_and_path):
                source_video_name_and_path = voice_fhd_file_name_and_path
            elif os.path.exists(viedo_file_name_and_path):
                logging.info(f"Cannot find high-definition video, use low-definition video {viedo_file_name_and_path} for preview video {preview_video_name_and_path}")
                source_video_name_and_path = viedo_file_name_and_path
            else:
                log_str = f"[WORK x] Error: Cannot find source video for preview video {preview_video_name_and_path}."
                logging.info(log_str)
                sys.exit(-1)

            logging.info(f"Generating zh preview video in {preview_video_name_and_path}")
            zh_video_preview(source_video_name_and_path, voice_connected_name_and_path, insturment_name_and_path, srt_voice_file_name_and_path, preview_video_name_and_path)
            logging.info(f"[WORK o] Generate zh preview video in {preview_video_name_and_path} successfully.")
        except Exception:
            log_str = f"[WORK x] Error: Program blocked while generating zh preview video in {preview_video_name_and_path}."
            logging.exception(log_str)
            sys.exit(-1)
    else:
        log_str = "[WORK -] Skip zh preview video."
        logging.info(log_str)

    logging.info("All done!!")
    logging.info("dir: " + work_path)


if __name__ == "__main__":
    main()
