"""
音频人声/伴奏分离模块
使用 Meta 的 Demucs v4 模型，支持 CUDA/MPS/CPU
模型首次运行时自动下载
"""
import logging
import os

import soundfile as sf
import torch


def _get_device() -> torch.device:
    """自动选择最优计算设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def audio_remove(
    audio_path: str,
    voice_output_path: str,
    instrument_output_path: str,
    model_name: str = "htdemucs",
) -> None:
    """
    使用 Demucs 分离音频中的人声和伴奏

    Args:
        audio_path: 输入音频文件路径
        voice_output_path: 人声输出路径
        instrument_output_path: 伴奏输出路径
        model_name: Demucs 模型名称（htdemucs / htdemucs_ft / mdx_extra）
    """
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import torchaudio

    device = _get_device()
    logging.info("音频分离使用设备: %s", device.type)

    # 加载模型（首次运行自动下载）
    logging.info("加载 Demucs 模型: %s", model_name)
    model = get_model(model_name)
    model.to(device)
    model.eval()
    logging.info("模型加载完成")

    # 加载音频
    waveform, sr = torchaudio.load(audio_path)
    logging.info("音频加载完成: %d Hz, %d 声道, %.1f 秒",
                 sr, waveform.shape[0], waveform.shape[1] / sr)

    # Demucs 需要的采样率
    if sr != model.samplerate:
        logging.info("重采样: %d Hz -> %d Hz", sr, model.samplerate)
        waveform = torchaudio.functional.resample(waveform, sr, model.samplerate)
        sr = model.samplerate

    # 确保是立体声
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # 添加 batch 维度: (channels, samples) -> (1, channels, samples)
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()
    waveform = waveform.unsqueeze(0).to(device)

    # 分离
    logging.info("开始音频分离...")
    with torch.no_grad():
        sources = apply_model(model, waveform, device=device, progress=True)

    # sources 形状: (1, num_sources, channels, samples)
    # Demucs 输出的源顺序: drums, bass, other, vocals
    sources = sources[0]
    sources = sources * ref.std() + ref.mean()

    # 找到 vocals 和其他源的索引
    source_names = model.sources
    vocals_idx = source_names.index("vocals")

    vocals = sources[vocals_idx].cpu().numpy().T
    # 伴奏 = 所有非人声源的混合
    accompaniment = sum(
        sources[i] for i in range(len(source_names)) if i != vocals_idx
    ).cpu().numpy().T

    # 保存
    sf.write(voice_output_path, vocals, sr)
    logging.info("人声已保存: %s", voice_output_path)

    sf.write(instrument_output_path, accompaniment, sr)
    logging.info("伴奏已保存: %s", instrument_output_path)
