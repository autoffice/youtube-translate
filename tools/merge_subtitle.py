"""
中英双语字幕合并工具
将中文和英文 SRT 字幕合并为 ASS 格式的双语字幕
"""
import logging

import pysubs2


def merge_subtitles(
    chinese_sub_path: str,
    english_sub_path: str,
    output_path: str,
    chinese_font_name: str = "Arial",
    chinese_font_size: int = 10,
    english_font_name: str = "Arial",
    english_font_size: int = 6,
) -> None:
    """
    合并中英文字幕为双语 ASS 字幕

    Args:
        chinese_sub_path: 中文字幕文件路径
        english_sub_path: 英文字幕文件路径
        output_path: 输出 ASS 文件路径
        chinese_font_name: 中文字体名称
        chinese_font_size: 中文字号
        english_font_name: 英文字体名称
        english_font_size: 英文字号
    """
    chinese_subs = pysubs2.load(chinese_sub_path)
    english_subs = pysubs2.load(english_sub_path)
    merged = pysubs2.SSAFile()

    merged.styles["ChineseStyle"] = pysubs2.SSAStyle(
        fontname=chinese_font_name,
        fontsize=chinese_font_size,
        primarycolor=pysubs2.Color(255, 255, 255),
        outlinecolor=pysubs2.Color(0, 0, 0),
        backcolor=pysubs2.Color(0, 0, 0),
        bold=True,
        marginv=10,
    )
    merged.styles["EnglishStyle"] = pysubs2.SSAStyle(
        fontname=english_font_name,
        fontsize=english_font_size,
        primarycolor=pysubs2.Color(255, 255, 255),
        outlinecolor=pysubs2.Color(0, 0, 0),
        backcolor=pysubs2.Color(0, 0, 0),
        marginv=10,
    )

    for i, zh_event in enumerate(chinese_subs):
        if i >= len(english_subs):
            break
        en_event = english_subs[i]
        event = pysubs2.SSAEvent(
            start=zh_event.start,
            end=zh_event.end,
            style="ChineseStyle",
        )
        event.text = (
            f"\\N{{\\fn{chinese_font_name}\\fs{chinese_font_size}}}{zh_event.text}"
            f"\\N{{\\fn{english_font_name}\\fs{english_font_size}\\i1}}{en_event.text}"
        )
        merged.append(event)

    merged.save(output_path)
    logging.info("双语字幕已保存: %s", output_path)
