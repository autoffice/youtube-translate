"""配置加载模块"""
import os
from typing import Dict

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


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
