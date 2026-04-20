from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from assignment2.modules.features import log_mel_spectrogram
from assignment2.utils.audio import signal_to_noise_ratio_db, slice_audio


def fgsm_attack(waveform: torch.Tensor, epsilon: float, gradient: torch.Tensor) -> torch.Tensor:
    return (waveform + epsilon * gradient.sign()).clamp(-1.0, 1.0)


@dataclass
class FGSMAttackTrial:
    epsilon: float
    snr_db: float
    target_frame_ratio: float
    success: bool


@dataclass
class FGSMAttackSearchResult:
    segment_start_sec: float
    segment_end_sec: float
    baseline_source_ratio: float
    successful_epsilon: float | None
    successful_waveform: torch.Tensor | None
    trials: list[FGSMAttackTrial]


def _lid_features(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
) -> torch.Tensor:
    features = log_mel_spectrogram(
        waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mean = features.mean(dim=1, keepdim=True)
    std = features.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (features - mean) / std


def _targeted_iterative_fgsm(
    model,
    waveform: torch.Tensor,
    epsilon: float,
    attack_steps: int,
    target_label: int,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
) -> torch.Tensor:
    if attack_steps <= 1:
        candidate = waveform.clone().detach().requires_grad_(True)
        feats = _lid_features(
            candidate,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        logits = model(feats)["language_logits"].squeeze(0)
        targets = torch.full((logits.size(0),), target_label, dtype=torch.long, device=logits.device)
        loss = -F.cross_entropy(logits, targets)
        model.zero_grad()
        loss.backward()
        return fgsm_attack(candidate.detach(), epsilon, candidate.grad.detach())

    step_size = epsilon / max(attack_steps, 1)
    adversarial = waveform.clone().detach()
    for _ in range(attack_steps):
        adversarial = adversarial.detach().requires_grad_(True)
        feats = _lid_features(
            adversarial,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        logits = model(feats)["language_logits"].squeeze(0)
        targets = torch.full((logits.size(0),), target_label, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, targets)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        adversarial = adversarial.detach() - step_size * gradient.sign()
        delta = (adversarial - waveform).clamp(-epsilon, epsilon)
        adversarial = (waveform + delta).clamp(-1.0, 1.0)
    return adversarial.detach()


def evaluate_fgsm_grid(
    model,
    waveform: torch.Tensor,
    sample_rate: int,
    target_label: int,
    epsilon_grid: list[float],
    snr_threshold_db: float,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    attack_steps: int = 1,
    success_ratio_threshold: float = 0.5,
) -> tuple[list[FGSMAttackTrial], torch.Tensor | None]:
    successful_waveform: torch.Tensor | None = None
    trials: list[FGSMAttackTrial] = []

    for epsilon in epsilon_grid:
        perturbed = _targeted_iterative_fgsm(
            model=model,
            waveform=waveform,
            epsilon=float(epsilon),
            attack_steps=attack_steps,
            target_label=target_label,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        snr_db = signal_to_noise_ratio_db(waveform, perturbed)

        with torch.inference_mode():
            adv_feats = _lid_features(
                perturbed,
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            adv_logits = model(adv_feats)["language_logits"].argmax(dim=-1).squeeze(0)

        target_frame_ratio = float((adv_logits == target_label).float().mean().item())
        success = snr_db > snr_threshold_db and target_frame_ratio >= success_ratio_threshold
        trials.append(
            FGSMAttackTrial(
                epsilon=float(epsilon),
                snr_db=float(snr_db),
                target_frame_ratio=target_frame_ratio,
                success=bool(success),
            )
        )
        if success and successful_waveform is None:
            successful_waveform = perturbed.detach()

    return trials, successful_waveform


def _baseline_source_ratio(
    model,
    waveform: torch.Tensor,
    source_label: int,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
) -> float:
    with torch.inference_mode():
        feats = _lid_features(
            waveform,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        probabilities = model(feats)["language_logits"].softmax(dim=-1).squeeze(0)
        predictions = probabilities.argmax(dim=-1)
    source_ratio = float((predictions == source_label).float().mean().item())
    target_prob = probabilities[:, 1 - source_label]
    return source_ratio, float(target_prob.mean().item()), float(target_prob.max().item())


def search_fgsm_attack(
    model,
    waveform: torch.Tensor,
    sample_rate: int,
    source_label: int,
    target_label: int,
    epsilon_grid: list[float],
    snr_threshold_db: float,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    segment_seconds: float = 5.0,
    stride_seconds: float = 2.5,
    max_candidate_segments: int = 8,
    attack_steps: int = 1,
    success_ratio_threshold: float = 0.5,
) -> FGSMAttackSearchResult | None:
    total_duration = waveform.size(-1) / sample_rate
    if total_duration <= 0:
        return None

    candidate_segments: list[tuple[float, float, float, float, float]] = []
    start_sec = 0.0
    while start_sec < max(total_duration - 1e-6, 0.0):
        end_sec = min(start_sec + segment_seconds, total_duration)
        if (end_sec - start_sec) < max(segment_seconds * 0.8, 1.0):
            break
        segment = slice_audio(waveform, sample_rate=sample_rate, start_sec=start_sec, end_sec=end_sec)
        source_ratio, target_prob_mean, target_prob_max = _baseline_source_ratio(
            model=model,
            waveform=segment,
            source_label=source_label,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        if source_ratio >= success_ratio_threshold:
            candidate_segments.append(
                (float(start_sec), float(end_sec), source_ratio, target_prob_mean, target_prob_max)
            )
        start_sec += max(stride_seconds, 0.5)

    if not candidate_segments:
        return None

    candidate_segments = sorted(candidate_segments, key=lambda item: (item[4], item[3], -item[2]), reverse=True)[
        : max_candidate_segments
    ]
    best_result: FGSMAttackSearchResult | None = None

    for start_sec, end_sec, source_ratio, _, _ in candidate_segments:
        segment = slice_audio(waveform, sample_rate=sample_rate, start_sec=start_sec, end_sec=end_sec)
        trials, successful_waveform = evaluate_fgsm_grid(
            model=model,
            waveform=segment,
            sample_rate=sample_rate,
            target_label=target_label,
            epsilon_grid=epsilon_grid,
            snr_threshold_db=snr_threshold_db,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            attack_steps=attack_steps,
            success_ratio_threshold=success_ratio_threshold,
        )
        successful_trial = next((trial for trial in trials if trial.success), None)
        current_result = FGSMAttackSearchResult(
            segment_start_sec=float(start_sec),
            segment_end_sec=float(end_sec),
            baseline_source_ratio=float(source_ratio),
            successful_epsilon=float(successful_trial.epsilon) if successful_trial is not None else None,
            successful_waveform=successful_waveform,
            trials=trials,
        )
        if best_result is None:
            best_result = current_result
            continue
        if current_result.successful_epsilon is not None:
            if best_result.successful_epsilon is None or current_result.successful_epsilon < best_result.successful_epsilon:
                best_result = current_result
        elif best_result.successful_epsilon is None:
            current_peak = max((trial.target_frame_ratio for trial in current_result.trials), default=0.0)
            best_peak = max((trial.target_frame_ratio for trial in best_result.trials), default=0.0)
            if current_peak > best_peak:
                best_result = current_result

    return best_result


def find_minimum_fgsm_epsilon(
    model,
    waveform: torch.Tensor,
    sample_rate: int,
    source_label: int,
    target_label: int,
    epsilon_grid: list[float],
    snr_threshold_db: float,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
) -> tuple[float | None, torch.Tensor | None]:
    result = search_fgsm_attack(
        model=model,
        waveform=waveform,
        sample_rate=sample_rate,
        source_label=source_label,
        target_label=target_label,
        epsilon_grid=epsilon_grid,
        snr_threshold_db=snr_threshold_db,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        segment_seconds=5.0,
        stride_seconds=2.5,
        max_candidate_segments=8,
        attack_steps=10,
        success_ratio_threshold=0.5,
    )
    if result is None:
        return None, None
    if result.successful_epsilon is not None:
        return result.successful_epsilon, result.successful_waveform
    return None, None
