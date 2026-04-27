"""使用阿里云 DashScope API 进行字幕翻译"""
import json
import logging
import os
import time
from typing import Dict, List, Optional

import requests
import tenacity
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "qwen3.5-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeTranslator:
    """使用阿里云 DashScope API 进行翻译的类"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 DASHSCOPE_API_KEY，请在 .env 文件中配置")
        self.model_name = model_name
        self.base_url = base_url
        self.terms: Dict[str, str] = {}

    def load_terms(self, terms_file: str) -> None:
        """加载术语表文件"""
        if not os.path.exists(terms_file):
            logging.warning("术语表文件不存在: %s", terms_file)
            return
        with open(terms_file, "r", encoding="utf-8") as f:
            self.terms = json.load(f)
        logging.info("已加载术语表，共 %d 条", len(self.terms))

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=5, max=30),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    def _request_api(self, system_text: str, user_text: str, max_tokens: int = 4096) -> dict:
        """调用 DashScope 兼容 API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": max_tokens,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    def translate_batch(self, texts: List[str]) -> List[str]:
        """批量翻译字幕文本（整体翻译，保留上下文）"""
        if not texts:
            return []

        start_time = time.time()
        system_text = (
            "你是一个专业的视频字幕翻译器，负责将英文字幕翻译为地道的中文字幕。\n"
            "翻译规则：\n"
            "1. 保持原文的语气和风格\n"
            "2. 使用简体中文\n"
            "3. 输入格式为编号加英文原文，每行一条\n"
            "4. 输出格式必须严格对应：每行一条翻译，编号和原文一一对应\n"
            "5. 不要添加任何解释、标记或额外内容\n"
            "6. 不要合并或拆分行，保持行数完全一致"
        )

        terms_hint = ""
        if self.terms:
            terms_hint = f"\n以下是术语翻译规则：\n{json.dumps(self.terms, ensure_ascii=False)}\n\n"

        numbered_lines = [f"{i+1}. {text}" for i, text in enumerate(texts)]
        input_text = "\n".join(numbered_lines)
        user_text = (
            f"{terms_hint}"
            f"请翻译以下 {len(texts)} 条字幕，每行对应输出一条中文翻译（保持编号格式）：\n\n"
            f"{input_text}"
        )

        logging.info("开始翻译 %d 条字幕...", len(texts))
        result = self._request_api(system_text=system_text, user_text=user_text, max_tokens=max(4096, len(texts) * 100))
        elapsed = time.time() - start_time
        raw_output = result["choices"][0]["message"]["content"].strip()
        translated = self._parse_numbered_output(raw_output, len(texts))
        logging.info("翻译完成，共 %d 条，耗时: %.2fs", len(texts), elapsed)
        return translated

    def _parse_numbered_output(self, raw_output: str, expected_count: int) -> List[str]:
        """解析带编号的翻译输出"""
        lines = [line.strip() for line in raw_output.strip().split("\n") if line.strip()]
        results = []
        for line in lines:
            stripped = line
            for sep in [". ", "、", "．", ") ", "） "]:
                dot_pos = line.find(sep)
                if dot_pos > 0 and line[:dot_pos].isdigit():
                    stripped = line[dot_pos + len(sep):]
                    break
            results.append(stripped)

        if len(results) < expected_count:
            logging.warning("翻译输出行数不足: 期望 %d 行，实际 %d 行", expected_count, len(results))
            results.extend([""] * (expected_count - len(results)))
        elif len(results) > expected_count:
            logging.warning("翻译输出行数过多: 期望 %d 行，实际 %d 行", expected_count, len(results))
            results = results[:expected_count]
        return results
