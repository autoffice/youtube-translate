# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 回复语言

总是用中文来回复。

## 项目概述

pytvzhen 是一个 YouTube 视频自动翻译配音流水线。它下载 YouTube 视频，提取音频，分离人声与背景音乐，通过 Whisper 将语音转写为英文字幕，使用阿里云 DashScope API 翻译为中文字幕，生成中文 TTS 语音，最终合成带中文配音和字幕的预览视频。

## 运行方式

```bash
# 安装
pip install -e .

# 运行
python -m youtube_translate
python -m youtube_translate -v VIDEO_ID
```

## 环境依赖

- Python 3.8+
- ffmpeg 必须安装并加入 PATH
- `pip install -e .`（或 `pip install -r requirements.txt`）
- 音频分离需要预训练模型文件（如 `models/audio_separation/baseline.pth`）
- 配置 `.env` 文件，设置 `DASHSCOPE_API_KEY`（参考 `.env.example`）

## 多平台支持

- **Windows**: 支持 NVIDIA GPU (CUDA)、AMD GPU、纯 CPU
- **macOS**: 支持 Apple Silicon (MPS)、Intel CPU、AMD GPU
- 自动检测最优计算设备，无需手动配置

## 架构

流水线由 `work_space.py:main()` 编排，通过 `.env` 环境变量控制。按顺序执行：

1. **下载视频** — `yt-dlp` 下载 YouTube 视频（标清 + 高清），自动检测已存在则跳过
2. **提取音频** — `moviepy` 从视频中提取 WAV 音频
3. **音频分离** — `tools/audio_remove.py` 使用神经网络将人声与伴奏分离（`lib/` 模块 — 基于 UNet 的模型，使用 librosa/torch）
4. **语音转写** — `faster_whisper`（CTranslate2）生成词级英文 SRT 字幕
5. **语句合并** — 将词级字幕重组为句级 SRT
6. **字幕翻译** — 使用阿里云 DashScope API（`tools/trans_dashscope.py`）翻译 SRT。支持术语文件（`tools/terms.json`）处理领域专用词汇
7. **文字转语音** — 通过 ChatTTS 本地模型将中文字幕转为语音，支持通过种子控制音色
8. **语音拼接** — 对齐并变速调整 TTS 片段以匹配原始时间轴，带交叉淡入淡出
9. **中文转写** — 对生成的中文语音重新转写，获取精确字幕时间轴
10. **双语字幕** — 可选生成中英双语 ASS 字幕
11. **预览合成** — FFmpeg 合成视频 + 中文配音 + 背景音乐 + 字幕

## 关键模块

- `lib/` — 音频分离神经网络（数据集加载、`nets.py` 中的模型架构、频谱工具）
- `tools/trans_dashscope.py` — `DashScopeTranslator`，封装阿里云 DashScope API 的批量翻译，带重试机制
- `tools/merge_subtitle.py` — 合并中英文字幕为双语 ASS 格式
- `tools/merge_video_srt.py` — FFmpeg 封装，用于向视频添加字幕和混音
- `tools/tts_chattts.py` — ChatTTS 封装，本地中文语音合成
- `tools/voice_redo.py` — 通过 ChatTTS 重新生成单条 TTS 片段
- `tools/audio_remove.py` — 音频人声/伴奏分离，支持多平台 GPU 加速

## 配置说明

所有配置通过 `.env` 文件管理（参考 `.env.example`）。主要配置项：

- `DASHSCOPE_API_KEY` — 阿里云 DashScope API 密钥
- `HTTP_PROXY` / `HTTPS_PROXY` — 代理地址（可选）
- `VIDEO_ID` — YouTube 视频 ID
- `OUTPUT_DIR` — 输出目录
- `AUDIO_SEPARATION_MODEL` — 人声分离模型权重路径
- `WHISPER_MODEL` / `WHISPER_ZH_MODEL` — Whisper 模型大小
- `TRANSLATE_MODEL` — DashScope 翻译模型名
- `DUAL_SUBTITLE` — 是否生成中英双语字幕
- `DUAL_ZH_FONT` / `DUAL_ZH_FONTSIZE` — 中文字幕样式
- `DUAL_EN_FONT` / `DUAL_EN_FONTSIZE` — 英文字幕样式
- `ENABLE_DUBBING` — 是否启用中文配音
- `TTS_SPEAKER_SEED` — ChatTTS 音色种子（0 表示随机）

## 代码规范

- 遵循 PEP 8 规范
- 使用类型注解
- 函数和类使用 docstring
- 变量命名使用 snake_case
- 常量使用 UPPER_CASE
