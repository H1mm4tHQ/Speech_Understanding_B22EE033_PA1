from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def load_audio(path: str | Path, target_sr: int | None = None, mono: bool = True) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    if mono and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if target_sr is not None and sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr
    return waveform, sample_rate


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform.detach().cpu(), sample_rate)


def peak_normalize(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    peak = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    return waveform / peak


def rms_normalize(waveform: torch.Tensor, target_rms: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    rms = waveform.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    return waveform * (target_rms / rms)


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def slice_audio(waveform: torch.Tensor, sample_rate: int, start_sec: float, end_sec: float) -> torch.Tensor:
    start = seconds_to_samples(start_sec, sample_rate)
    end = seconds_to_samples(end_sec, sample_rate)
    return waveform[..., start:end]


def chunk_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float = 0.0,
) -> list[tuple[float, float, torch.Tensor]]:
    chunk_size = seconds_to_samples(chunk_seconds, sample_rate)
    hop_size = seconds_to_samples(max(chunk_seconds - overlap_seconds, 1e-6), sample_rate)
    total = waveform.size(-1)
    chunks: list[tuple[float, float, torch.Tensor]] = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        chunk = waveform[..., start:end]
        chunks.append((start / sample_rate, end / sample_rate, chunk))
        if end == total:
            break
        start += hop_size
    return chunks


def unfold_frames(waveform: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
    return waveform.unfold(0, frame_length, hop_length)


def frame_rms(waveform: torch.Tensor, frame_length: int, hop_length: int, eps: float = 1e-8) -> torch.Tensor:
    frames = unfold_frames(waveform, frame_length, hop_length)
    return frames.pow(2).mean(dim=-1).sqrt().clamp_min(eps)


def signal_to_noise_ratio_db(clean: torch.Tensor, perturbed: torch.Tensor, eps: float = 1e-8) -> float:
    noise = perturbed - clean
    signal_power = clean.pow(2).mean().clamp_min(eps)
    noise_power = noise.pow(2).mean().clamp_min(eps)
    return float(10.0 * torch.log10(signal_power / noise_power))

