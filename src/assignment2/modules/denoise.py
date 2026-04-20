from __future__ import annotations

import torch
import torch.nn as nn


class SpectralSubtractionDenoiser(nn.Module):
    """Classical spectral subtraction for classroom denoising."""

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        alpha: float = 1.5,
        floor: float = 0.02,
        noise_estimate_frames: int = 60,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.alpha = alpha
        self.floor = floor
        self.noise_estimate_frames = noise_estimate_frames
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        magnitude = spec.abs()
        phase = torch.angle(spec)
        noise_mag = magnitude[..., : self.noise_estimate_frames].mean(dim=-1, keepdim=True)
        clean_mag = torch.clamp(magnitude - self.alpha * noise_mag, min=self.floor * noise_mag)
        clean_spec = torch.polar(clean_mag, phase)
        clean_waveform = torch.istft(
            clean_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=waveform.size(-1),
        )
        return clean_waveform.unsqueeze(0) if clean_waveform.dim() == 1 else clean_waveform

