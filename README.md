# YouTube 视频自动翻译并配音

一个自动化的 YouTube 视频翻译配音工具，支持下载视频、提取音频、分离人声、语音转写、字幕翻译、TTS 配音和视频合成。

## 功能特性

- ✅ 自动下载 YouTube 视频（支持标清和高清）
- ✅ 音频人声/伴奏分离（基于深度学习）
- ✅ 英文语音转写（Whisper）
- ✅ 字幕翻译（阿里云 DashScope API）
- ✅ 中英双语字幕生成
- ✅ 文字转语音（ChatTTS 本地模型）
- ✅ 智能语音拼接（自动变速对齐）
- ✅ 视频合成（FFmpeg）
- ✅ 多平台支持（Windows/macOS，CUDA/MPS/CPU）

## 环境要求

- Python 3.8+
- ffmpeg（必须安装并加入 PATH）
- 阿里云 DashScope API Key

## 安装

1. 克隆仓库

```bash
git clone https://github.com/your-repo/pytvzhen.git
cd pytvzhen
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 配置环境变量

复制 `.env.example` 为 `.env`，并填入你的 DashScope API Key：

```bash
cp .env.example .env
# 编辑 .env 文件，设置 DASHSCOPE_API_KEY
```

4. 下载音频分离模型

将预训练模型放置到 `models/audio_separation/baseline.pth`

## 使用方法

### 基本用法

```bash
# 使用 .env 中的默认配置运行
python work_space.py

# 覆盖视频 ID
python work_space.py -v VIDEO_ID
```

### 配置说明

所有配置项都在 `.env` 文件中管理，复制 `.env.example` 为 `.env` 后按需修改：

```bash
# 阿里云 DashScope API 配置
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# 代理配置（可选）
# HTTP_PROXY=http://127.0.0.1:7890
# HTTPS_PROXY=http://127.0.0.1:7890

# 视频配置
VIDEO_ID=iMVLoZaJ_1M
OUTPUT_DIR=./output

# 模型配置
AUDIO_SEPARATION_MODEL=models/audio_separation/baseline.pth
WHISPER_MODEL=medium
WHISPER_ZH_MODEL=medium
TRANSLATE_MODEL=qwen3.5-plus

# 双语字幕配置
DUAL_SUBTITLE=true
DUAL_ZH_FONT=Arial
DUAL_ZH_FONTSIZE=10
DUAL_EN_FONT=Arial
DUAL_EN_FONTSIZE=8

# 中文配音配置
ENABLE_DUBBING=false
TTS_SPEAKER_SEED=42
```

## 工作流程

1. **下载视频** → 从 YouTube 下载视频（自动检测，已存在则跳过）
2. **提取音频** → 从视频中提取音频
3. **音频分离** → 分离人声和背景音乐
4. **语音转写** → 将英文语音转为字幕
5. **字幕合并** → 将词级字幕合并为句级
6. **字幕翻译** → 使用 DashScope API 翻译为中文
7. **字幕转语音** → 使用 ChatTTS 生成中文配音
8. **语音拼接** → 拼接并对齐语音片段
9. **中文转写** → 重新转写中文语音（可选）
10. **双语字幕** → 生成中英双语字幕（可选）
11. **视频合成** → 合成最终预览视频

## 多平台支持

- **Windows**
  - NVIDIA GPU (CUDA)
  - AMD GPU
  - CPU
- **macOS**
  - Apple Silicon (MPS)
  - Intel CPU
  - AMD GPU

程序会自动检测并使用最优计算设备。

## 工具脚本

### 重新生成语音

如果某条字幕的语音质量不佳，可以使用 `tools/voice_redo.py` 重新生成：

```bash
python tools/voice_redo.py
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
