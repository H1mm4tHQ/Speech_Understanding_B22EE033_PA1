from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio

from assignment2.modules.features import normalize_contour, pitch_and_energy


@dataclass
class ProsodyProfile:
    pitch: torch.Tensor
    energy: torch.Tensor


def extract_prosody(
    waveform: torch.Tensor,
    sample_rate: int,
    hop_ms: float,
    frame_length: int,
    hop_length: int,
) -> ProsodyProfile:
    pitch, energy = pitch_and_energy(
        waveform,
        sample_rate=sample_rate,
        hop_ms=hop_ms,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return ProsodyProfile(pitch=pitch.squeeze(0), energy=energy.squeeze(0))


def dtw_path(source: torch.Tensor, target: torch.Tensor) -> list[tuple[int, int]]:
    cost = torch.cdist(source, target, p=2)
    dp = torch.full((source.size(0) + 1, target.size(0) + 1), float("inf"))
    dp[0, 0] = 0.0
    backptr: dict[tuple[int, int], tuple[int, int]] = {}

    for i in range(1, source.size(0) + 1):
        for j in range(1, target.size(0) + 1):
            candidates = [
                (dp[i - 1, j], (i - 1, j)),
                (dp[i, j - 1], (i, j - 1)),
                (dp[i - 1, j - 1], (i - 1, j - 1)),
            ]
            best_cost, best_prev = min(candidates, key=lambda item: item[0])
            dp[i, j] = cost[i - 1, j - 1] + best_cost
            backptr[(i, j)] = best_prev

    i, j = source.size(0), target.size(0)
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = backptr[(i, j)]
    path.reverse()
    return path


def warp_prosody(source: ProsodyProfile, target: ProsodyProfile) -> ProsodyProfile:
    source_stack = torch.stack(
        [normalize_contour(source.pitch), normalize_contour(source.energy)],
        dim=-1,
    )
    target_stack = torch.stack(
        [normalize_contour(target.pitch), normalize_contour(target.energy)],
        dim=-1,
    )
    path = dtw_path(source_stack, target_stack)

    warped_pitch = torch.zeros_like(target.pitch)
    warped_energy = torch.zeros_like(target.energy)
    counts = torch.zeros_like(target.pitch)

    for src_idx, tgt_idx in path:
        warped_pitch[tgt_idx] += source.pitch[src_idx]
        warped_energy[tgt_idx] += source.energy[src_idx]
        counts[tgt_idx] += 1

    counts = counts.clamp_min(1.0)
    return ProsodyProfile(pitch=warped_pitch / counts, energy=warped_energy / counts)


def apply_prosody_warp(
    waveform: torch.Tensor,
    sample_rate: int,
    original_target: ProsodyProfile,
    warped_target: ProsodyProfile,
    hop_length: int,
) -> torch.Tensor:
    pitch_ratio = warped_target.pitch.median().clamp_min(1e-4) / original_target.pitch.median().clamp_min(1e-4)
    semitones = int(torch.round(12.0 * torch.log2(pitch_ratio)).item())
    semitones = max(-2, min(2, semitones))
    shifted = waveform
    if semitones != 0:
        try:
            shifted = torchaudio.functional.pitch_shift(
                waveform,
                sample_rate=sample_rate,
                n_steps=semitones,
                hop_length=hop_length,
            )
        except RuntimeError:
            shifted = waveform

    frame_gain = warped_target.energy / original_target.energy.clamp_min(1e-4)
    gain_samples = torch.nn.functional.interpolate(
        frame_gain.view(1, 1, -1),
        size=shifted.size(-1),
        mode="linear",
        align_corners=False,
    ).squeeze(0)
    return shifted * gain_samples
