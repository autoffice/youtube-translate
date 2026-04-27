"""
使用阿里云 DashScope API 进行字幕翻译的工具类
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests
import tenacity
from dotenv import load_dotenv
import os


# 加载环境变量
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
        """
        初始化翻译器

        Args:
            api_key: DashScope API 密钥，如果为 None 则从环境变量读取
            model_name: 使用的模型名称
            base_url: API 基础 URL
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 DASHSCOPE_API_KEY，请在 .env 文件中配置")

        self.model_name = model_name
        self.base_url = base_url
        self.terms: Dict[str, str] = {}

    def load_terms(self, terms_file: str) -> None:
        """
        加载术语表文件

        Args:
            terms_file: 术语表 JSON 文件路径
        """
        if not os.path.exists(terms_file):
            logging.warning("术语表文件不存在: %s", terms_file)
            return
        with open(terms_file, "r", encoding="utf-8") as f:
            self.terms = json.load(f)
        logging.info("已加载术语表，共 %d 条", len(self.terms))

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=3, max=30),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def _request_api(
        self,
        system_text: str,
        user_text: str,
        max_tokens: int = 2000,
    ) -> dict:
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
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def translate_text(self, text: str, max_tokens: int = 2000) -> dict:
        """
        翻译单条文本

        Args:
            text: 待翻译的英文文本
            max_tokens: 最大生成 token 数

        Returns:
            包含翻译结果和耗时的字典
        """
        start_time = time.time()

        system_text = (
            "你是一个专业的视频字幕翻译器，负责将英文字幕翻译为地道的中文字幕。"
            "翻译时请注意：\n"
            "1. 保持原文的语气和风格\n"
            "2. 使用简体中文\n"
            "3. 只输出翻译结果，不要添加任何解释或标记"
        )

        terms_hint = ""
        if self.terms:
            terms_hint = f"\n\n以下是需要遵循的术语翻译规则：\n{json.dumps(self.terms, ensure_ascii=False)}\n\n"

        user_text = f"{terms_hint}请翻译以下字幕文本：\n{text}"

        result = self._request_api(
            system_text=system_text,
            user_text=user_text,
            max_tokens=max_tokens,
        )

        elapsed = time.time() - start_time
        text_result = result["choices"][0]["message"]["content"].strip()

        # 清理可能的 markdown 代码块包裹
        if "```" in text_result:
            parts = text_result.split("```")
            if len(parts) >= 3:
                text_result = parts[1].strip()
            elif len(parts) >= 2:
                text_result = parts[1].strip()

        return {
            "text_result": text_result,
            "model": result.get("model", self.model_name),
            "time": elapsed,
        }

    def translate_batch(
        self,
        texts: List[str],
        max_tokens: int = 2000,
        max_workers: int = 5,
    ) -> List[str]:
        """
        批量翻译文本

        Args:
            texts: 待翻译的文本列表
            max_tokens: 每条翻译的最大 token 数
            max_workers: 最大并发线程数

        Returns:
            翻译结果列表，顺序与输入一致
        """
        results: List[Optional[str]] = [None] * len(texts)
        failed_indices = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.translate_text, text, max_tokens): i
                for i, text in enumerate(texts)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = result["text_result"]
                    logging.info(
                        "翻译完成 [%d/%d] 耗时: %.2fs",
                        idx + 1, len(texts), result["time"],
                    )
                except Exception:
                    logging.exception("翻译失败 [%d/%d]", idx + 1, len(texts))
                    failed_indices.append(idx)
                    results[idx] = texts[idx]

        # 对失败的条目进行逐条重试
        if failed_indices:
            logging.info("正在重试 %d 条失败的翻译...", len(failed_indices))
            for idx in failed_indices:
                try:
                    result = self.translate_text(texts[idx], max_tokens)
                    results[idx] = result["text_result"]
                    logging.info("重试成功 [%d/%d]", idx + 1, len(texts))
                except Exception:
                    logging.error("重试仍然失败 [%d/%d]，保留原文", idx + 1, len(texts))

        return results
