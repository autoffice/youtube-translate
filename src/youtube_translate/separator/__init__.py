"""音频人声/伴奏分离模块"""
import logging

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from youtube_translate.separator import dataset, nets, spec_utils

FFT_SIZE = 2048
HOP_SIZE = 1024


def _get_device() -> torch.device:
    """自动选择最优计算设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Separator:
    def __init__(self, model, device: torch.device, batchsize: int = 4, cropsize: int = 256):
        self.model = model
        self.offset = model.offset
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize

    def _separate(self, X_spec_pad: np.ndarray, roi_size: int) -> np.ndarray:
        patches = (X_spec_pad.shape[2] - 2 * self.offset) // roi_size
        X_dataset = np.asarray([
            X_spec_pad[:, :, i * roi_size: i * roi_size + self.cropsize]
            for i in range(patches)
        ])

        self.model.eval()
        mask_list = []
        with torch.no_grad():
            for i in tqdm(range(0, patches, self.batchsize)):
                batch = torch.from_numpy(X_dataset[i: i + self.batchsize]).to(self.device)
                mask = self.model.predict_mask(torch.abs(batch))
                mask_list.append(np.concatenate(mask.detach().cpu().numpy(), axis=2))
        return np.concatenate(mask_list, axis=2)

    def _postprocess(self, X_spec: np.ndarray, mask: np.ndarray):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)
        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        return y_spec, v_spec

    def separate_tta(self, X_spec: np.ndarray):
        """使用 TTA（测试时增强）提升分离质量"""
        n_frame = X_spec.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)

        X_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
        X_pad /= X_pad.max()
        mask = self._separate(X_pad, roi_size)

        pad_l2 = pad_l + roi_size // 2
        pad_r2 = pad_r + roi_size // 2
        X_pad2 = np.pad(X_spec, ((0, 0), (0, 0), (pad_l2, pad_r2)), mode="constant")
        X_pad2 /= X_pad2.max()
        mask_tta = self._separate(X_pad2, roi_size)
        mask_tta = mask_tta[:, :, roi_size // 2:]

        mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        return self._postprocess(X_spec, mask)


def audio_remove(
    audio_path: str,
    voice_output_path: str,
    instrument_output_path: str,
    model_path: str,
) -> None:
    """
    分离音频中的人声和伴奏

    Args:
        audio_path: 输入音频文件路径
        voice_output_path: 人声输出路径
        instrument_output_path: 伴奏输出路径
        model_path: 模型权重文件路径
    """
    device = _get_device()
    logging.info("音频分离使用设备: %s", device.type)

    model = nets.CascadedNet(FFT_SIZE, HOP_SIZE, 32, 128)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    logging.info("模型加载完成")

    X, sr = librosa.load(audio_path, sr=None, mono=False, dtype=np.float32, res_type="kaiser_fast")
    if X.ndim == 1:
        X = np.stack([X, X])

    # 如果采样率不是 44.1kHz，重采样（模型训练时使用的采样率）
    if sr != 44100:
        logging.info("重采样音频: %d Hz -> 44100 Hz", sr)
        X = librosa.resample(X, orig_sr=sr, target_sr=44100, res_type="kaiser_fast")
        sr = 44100

    X_spec = spec_utils.wave_to_spectrogram(X, HOP_SIZE, FFT_SIZE)
    separator = Separator(model=model, device=device)
    y_spec, v_spec = separator.separate_tta(X_spec)

    sf.write(instrument_output_path, spec_utils.spectrogram_to_wave(y_spec, HOP_SIZE).T, sr)
    logging.info("伴奏已保存: %s", instrument_output_path)

    sf.write(voice_output_path, spec_utils.spectrogram_to_wave(v_spec, HOP_SIZE).T, sr)
    logging.info("人声已保存: %s", voice_output_path)
