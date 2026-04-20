from __future__ import annotations

import math

import torch
import torchaudio

from assignment2.utils.audio import frame_rms


def log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
) -> torch.Tensor:
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel = transform(waveform)
    return torch.log(mel.clamp_min(1e-6)).transpose(1, 2)


def lfcc_features(
    waveform: torch.Tensor,
    sample_rate: int,
    n_lfcc: int = 40,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
) -> torch.Tensor:
    transform = torchaudio.transforms.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        speckwargs={
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "center": False,
        },
    )
    feats = transform(waveform)
    return feats.transpose(1, 2)


def mfcc_features(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mfcc: int = 40,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
) -> torch.Tensor:
    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "n_mels": n_mels,
        },
    )
    feats = transform(waveform)
    return feats.transpose(1, 2)


def pitch_and_energy(
    waveform: torch.Tensor,
    sample_rate: int,
    hop_ms: float,
    frame_length: int,
    hop_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pitch = torchaudio.functional.detect_pitch_frequency(
        waveform,
        sample_rate=sample_rate,
        frame_time=hop_ms / 1000.0,
    )
    energy = frame_rms(waveform, frame_length=frame_length, hop_length=hop_length)
    min_frames = min(pitch.size(-1), energy.size(-1))
    return pitch[..., :min_frames], energy[..., :min_frames]


def normalize_contour(sequence: torch.Tensor) -> torch.Tensor:
    mean = sequence.mean(dim=-1, keepdim=True)
    std = sequence.std(dim=-1, keepdim=True).clamp_min(1e-8)
    return (sequence - mean) / std


def mel_cepstral_distortion(mfcc_a: torch.Tensor, mfcc_b: torch.Tensor) -> float:
    min_frames = min(mfcc_a.size(0), mfcc_b.size(0))
    a = mfcc_a[:min_frames]
    b = mfcc_b[:min_frames]
    diff = a[:, 1:] - b[:, 1:]
    value = (10.0 / math.log(10.0)) * math.sqrt(2.0) * diff.pow(2).sum(dim=-1).sqrt().mean().item()
    return float(value)
