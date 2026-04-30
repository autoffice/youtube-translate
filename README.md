# YouTube 视频自动翻译并配音

一个自动化的 YouTube 视频翻译配音工具，支持下载视频、提取音频、分离人声、语音转写、字幕翻译、TTS 配音、视频合成和 B 站上传。

## 功能特性

- ✅ 自动下载 YouTube 视频（支持标清和高清）
- ✅ 音频人声/伴奏分离（基于深度学习）
- ✅ 英文语音转写（Whisper）
- ✅ 字幕翻译（阿里云 DashScope API）
- ✅ AI 自动生成视频标题、标签、描述
- ✅ 中英双语字幕生成
- ✅ 文字转语音（ChatTTS 本地模型）
- ✅ 智能语音拼接（自动变速对齐）
- ✅ 视频合成（FFmpeg）
- ✅ 自动生成带标题的封面图
- ✅ 一键上传到 B 站（支持草稿/发布）
- ✅ 多平台支持（Windows/macOS，CUDA/MPS/CPU）

## 环境要求

- Python 3.8+
- ffmpeg（必须安装并加入 PATH）
- 阿里云 DashScope API Key

## 安装

1. 克隆仓库

```bash
git clone https://github.com/your-repo/youtube-translate.git
cd youtube-translate
```

2. 安装依赖

```bash
pip install -r requirements.txt
# 或以开发模式安装
pip install -e .
```

3. 配置环境变量

复制 `.env.example` 为 `.env`，并填入你的配置：

```bash
cp .env.example .env
# 编辑 .env 文件，至少设置 DASHSCOPE_API_KEY
```

4. 下载音频分离模型

将预训练模型放置到 `models/audio_separation/baseline.pth`

## 使用方法

### 基本用法

```bash
# 使用 .env 中的默认配置运行
python -m youtube_translate

# 指定视频 ID
python -m youtube_translate -v VIDEO_ID

# 或使用命令行工具（安装后可用）
youtube-translate -v VIDEO_ID

# 使用批处理脚本（Windows）
yt VIDEO_ID
```

### 配置说明

所有配置项都在 `.env` 文件中管理，复制 `.env.example` 为 `.env` 后按需修改：

```bash
# ============================================
# 阿里云 DashScope API 配置
# ============================================
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# ============================================
# 代理配置（可选）
# ============================================
# DOWNLOAD_PROXY=http://127.0.0.1:7890

# ============================================
# 视频配置
# ============================================
VIDEO_ID=iMVLoZaJ_1M
OUTPUT_DIR=./output

# ============================================
# 模型配置
# ============================================
AUDIO_SEPARATION_MODEL=models/audio_separation/baseline.pth
WHISPER_MODEL=medium
WHISPER_ZH_MODEL=medium
TRANSLATE_MODEL=qwen3.5-plus

# ============================================
# 双语字幕配置
# ============================================
DUAL_SUBTITLE=true
DUAL_ZH_FONT=Arial
DUAL_ZH_FONTSIZE=10
DUAL_EN_FONT=Arial
DUAL_EN_FONTSIZE=8

# ============================================
# 中文配音配置
# ============================================
ENABLE_DUBBING=false
TTS_SPEAKER_SEED=42

# ============================================
# B 站上传配置
# ============================================
BILIBILI_UPLOAD=false
BILIBILI_PUBLISH=false
BILIBILI_COPYRIGHT=1
BILIBILI_TID=188
# BILIBILI_COOKIE=cookies.json
```

### B 站上传配置

如需上传到 B 站，需要先获取 Cookie：

```bash
# 安装 biliup
pip install biliup

# 登录获取 Cookie（会打开浏览器扫码）
biliup login

# 在 .env 中配置
BILIBILI_UPLOAD=true
BILIBILI_COOKIE=cookies.json
```

## 工作流程

1. **下载视频** → 从 YouTube 下载视频（自动检测，已存在则跳过）
2. **提取音频** → 使用 FFmpeg 从视频中提取音频
3. **音频分离** → 分离人声和背景音乐
4. **语音转写** → 将英文语音转为字幕
5. **字幕合并** → 将词级字幕合并为句级
6. **字幕翻译** → 使用 DashScope API 翻译为中文
7. **生成元数据** → AI 自动生成视频标题、标签、描述
8. **字幕转语音** → 使用 ChatTTS 生成中文配音（可选）
9. **语音拼接** → 拼接并对齐语音片段（可选）
10. **中文转写** → 重新转写中文语音（可选）
11. **双语字幕** → 生成中英双语字幕（可选）
12. **视频合成** → 合成最终视频
13. **生成封面** → 从视频截帧并叠加标题文字
14. **上传 B 站** → 自动上传到 B 站（可选）

## 输出文件

每个视频处理完成后，会在 `output/{VIDEO_ID}/` 目录下生成：

- `{VIDEO_ID}.mp4` — 原始下载的视频
- `{VIDEO_ID}_fhd.mp4` — 高清视频（如果下载成功）
- `{VIDEO_ID}.wav` — 提取的音频
- `{VIDEO_ID}_voice.wav` — 分离的人声
- `{VIDEO_ID}_instrument.wav` — 分离的伴奏
- `{VIDEO_ID}_en.srt` — 英文字幕（词级）
- `{VIDEO_ID}_en_merge.srt` — 英文字幕（句级）
- `{VIDEO_ID}_zh_merge.srt` — 中文字幕
- `{VIDEO_ID}_metadata.json` — AI 生成的标题、标签、描述
- `{VIDEO_ID}_cover.jpg` — 带标题的封面图
- `{VIDEO_ID}_output.mp4` — 最终合成的视频

## 项目结构

```
youtube-translate/
├── src/youtube_translate/       # 主包
│   ├── __init__.py
│   ├── __main__.py              # 命令行入口
│   ├── config.py                # 配置加载
│   ├── pipeline.py              # 主流程编排
│   ├── transcriber.py           # Whisper 语音转写
│   ├── translator.py            # DashScope 翻译和元数据生成
│   ├── subtitle.py              # 双语字幕合并
│   ├── tts.py                   # ChatTTS 语音合成
│   ├── video.py                 # 视频下载和合成
│   ├── uploader.py              # B 站上传
│   ├── separator/               # 音频分离模块
│   │   ├── __init__.py
│   │   ├── nets.py              # 神经网络模型
│   │   ├── layers.py
│   │   ├── spec_utils.py
│   │   └── ...
│   └── resources/
│       └── terms.json           # 术语表
├── scripts/
│   └── voice_redo.py            # 重新生成单条语音
├── pyproject.toml               # 包配置
├── requirements.txt             # 依赖列表
├── .env.example                 # 配置模板
└── README.md
```

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

如果某条字幕的语音质量不佳，可以使用 `scripts/voice_redo.py` 重新生成：

```bash
python scripts/voice_redo.py
```

## 常见问题

### 1. 如何获取 B 站 Cookie？

```bash
pip install biliup
biliup login
```

登录成功后会在当前目录生成 `cookies.json`，在 `.env` 中配置路径即可。

### 2. 视频上传失败怎么办？

- 检查 Cookie 是否过期（重新运行 `biliup login`）
- 检查视频文件是否完整
- 检查网络连接

### 3. 如何修改封面字体？

封面默认使用黑体（SimHei），字体路径硬编码在 `pipeline.py` 的 `_generate_cover()` 函数中。如需修改，编辑该函数中的 `fontfile` 参数。

### 4. 翻译质量不好怎么办？

- 在 `src/youtube_translate/resources/terms.json` 中添加术语表
- 调整 `TRANSLATE_MODEL` 使用更强的模型（如 `qwen-max`）

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
