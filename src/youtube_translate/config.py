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
        "BILIBILI_UPLOAD": os.getenv("BILIBILI_UPLOAD", "false").lower() == "true",
        "BILIBILI_PUBLISH": os.getenv("BILIBILI_PUBLISH", "false").lower() == "true",
        "BILIBILI_COPYRIGHT": int(os.getenv("BILIBILI_COPYRIGHT", "1")),
        "BILIBILI_TID": int(os.getenv("BILIBILI_TID", "188")),
        "BILIBILI_COOKIE": os.getenv("BILIBILI_COOKIE", ""),
    }
