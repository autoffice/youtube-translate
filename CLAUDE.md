# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 回复语言

总是用中文来回复。

## 项目概述

youtube-translate 是一个 YouTube 视频自动翻译配音工具。它下载 YouTube 视频，提取音频，使用 Meta Demucs v4 分离人声与背景音乐，通过 Whisper 将语音转写为英文字幕，使用阿里云 DashScope API 翻译为中文字幕并生成视频元数据（标题/标签/描述），可选生成中文 TTS 配音，最终合成带字幕的视频，并支持自动上传到 B 站。

## 运行方式

```bash
# 安装
pip install -e .

# 运行
python -m youtube_translate
python -m youtube_translate -v VIDEO_ID
youtube-translate -v VIDEO_ID
```

## 环境依赖

- Python 3.8+
- ffmpeg 必须安装并加入 PATH
- `pip install -e .`（或 `pip install -r requirements.txt`）
- 配置 `.env` 文件，设置 `DASHSCOPE_API_KEY`（参考 `.env.example`）
- Demucs 模型首次运行时自动下载（约 300MB）

## 多平台支持

- **Windows**: 支持 NVIDIA GPU (CUDA)、AMD GPU、纯 CPU
- **macOS**: 支持 Apple Silicon (MPS)、Intel CPU、AMD GPU
- 自动检测最优计算设备，无需手动配置

## 架构

流水线由 `src/youtube_translate/pipeline.py:main()` 编排，通过 `.env` 环境变量控制。所有步骤自动检测输出文件是否存在，已存在则跳过。按顺序执行：

1. **下载视频** — `yt-dlp` 下载 YouTube 视频（标清 + 高清），已存在则跳过
2. **提取音频** — `ffmpeg` 从视频中提取 48kHz 立体声 WAV 音频
3. **音频分离** — Meta Demucs v4 (htdemucs) 使用深度学习分离人声与伴奏，支持 CUDA/MPS/CPU 加速
4. **语音转写** — `openai-whisper` 生成英文 SRT 字幕
5. **字幕翻译** — 使用阿里云 DashScope API（`translator.py`）整体翻译 SRT，保留上下文。支持术语文件（`resources/terms.json`）
6. **生成元数据** — AI 根据字幕内容自动生成视频标题、标签、描述，保存为 `_metadata.json`
7. **中文配音**（可选）— 通过 ChatTTS 本地模型将中文字幕转为语音并自动拼接
8. **双语字幕**（可选）— 生成中英双语 ASS 字幕
9. **视频合成** — FFmpeg 合成视频 + 字幕（+ 中文配音 + 背景音乐）
10. **生成封面** — 从视频截帧并用 ffmpeg drawtext 叠加黄色标题文字（黑色阴影）
11. **上传 B 站**（可选）— 使用 biliup 上传视频到 B 站

## 项目结构

```
src/youtube_translate/
├── __init__.py          # 包版本信息
├── __main__.py          # python -m youtube_translate 入口
├── config.py            # 从 .env 加载配置
├── pipeline.py          # 主流程编排（main 函数）
├── transcriber.py       # Whisper 英文/中文转写、字幕合并
├── translator.py        # DashScope 翻译和视频元数据生成
├── subtitle.py          # 中英双语字幕合并（ASS 格式）
├── tts.py               # ChatTTS 语音合成
├── video.py             # 视频下载（yt-dlp）和合成（ffmpeg）
├── uploader.py          # B 站上传（biliup）
├── separator/           # 音频人声/伴奏分离（Demucs v4）
│   └── __init__.py      # audio_remove 函数，封装 Demucs API
└── resources/
    └── terms.json       # 翻译术语表
```

## 配置说明

所有配置通过 `.env` 文件管理（参考 `.env.example`）。主要配置项：

- `DASHSCOPE_API_KEY` — 阿里云 DashScope API 密钥
- `DOWNLOAD_PROXY` — 视频下载代理（可选，仅用于 YouTube）
- `VIDEO_ID` — YouTube 视频 ID
- `OUTPUT_DIR` — 输出根目录（实际输出到 `{OUTPUT_DIR}/{VIDEO_ID}/`）
- `WHISPER_MODEL` / `WHISPER_ZH_MODEL` — Whisper 模型大小
- `TRANSLATE_MODEL` — DashScope 翻译模型名
- `DUAL_SUBTITLE` — 是否生成中英双语字幕
- `DUAL_ZH_FONT` / `DUAL_ZH_FONTSIZE` — 中文字幕样式
- `DUAL_EN_FONT` / `DUAL_EN_FONTSIZE` — 英文字幕样式
- `ENABLE_DUBBING` — 是否启用中文配音
- `TTS_SPEAKER_SEED` — ChatTTS 音色种子（0 表示随机）
- `BILIBILI_UPLOAD` — 是否上传到 B 站
- `BILIBILI_PUBLISH` — 是否直接发布（false=草稿）
- `BILIBILI_COPYRIGHT` — 版权类型（1=原创，2=转载）
- `BILIBILI_TID` — B 站分区 ID（188=科普人文）
- `BILIBILI_COOKIE` — B 站 Cookie 文件路径

## 代码规范

- 遵循 PEP 8 规范
- 使用类型注解
- 函数和类使用 docstring
- 变量命名使用 snake_case
- 常量使用 UPPER_CASE
