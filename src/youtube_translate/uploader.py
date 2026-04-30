"""B 站视频上传模块"""
import json
import logging
from typing import Optional


def upload_to_bilibili(
    video_path: str,
    cover_path: str,
    metadata_path: str,
    cookie_file: str,
    tid: int = 188,
    copyright: int = 1,
    source: str = "",
    publish: bool = False,
) -> None:
    """
    上传视频到 B 站

    Args:
        video_path: 视频文件路径
        cover_path: 封面图片路径
        metadata_path: 元数据 JSON 文件路径（包含 title/tags/desc）
        cookie_file: B 站 Cookie 文件路径
        tid: 分区 ID
        copyright: 版权类型（1=自制原创，2=转载）
        source: 转载来源 URL（copyright=2 时必填）
        publish: True=直接发布，False=存为草稿
    """
    try:
        from biliup.plugins.bili_webup import BiliBili, Data
    except ImportError:
        logging.error("biliup 未安装，请运行: pip install biliup")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    title = metadata.get("title", "翻译视频")
    tags = [t.strip() for t in metadata.get("tags", "翻译").split(",")]
    desc = metadata.get("desc", "")

    video = Data()
    video.title = title
    video.desc = desc
    video.tag = tags
    video.tid = tid
    video.copyright = copyright
    if copyright == 2 and source:
        video.source = source
    video.cover = cover_path

    if not publish:
        video.dtime = 0  # 草稿模式

    with BiliBili(video) as bili:
        bili.login_by_cookies(cookie_file)
        video_part = bili.upload_file(video_path)
        video.append(video_part)
        ret = bili.submit()

    status = "发布" if publish else "草稿"
    logging.info("B 站上传完成（%s）: %s", status, title)
    logging.info("上传结果: %s", ret)
