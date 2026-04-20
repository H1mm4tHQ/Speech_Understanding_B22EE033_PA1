from __future__ import annotations

import re

import torch


WORD_PATTERN = re.compile(r"[A-Za-z0-9_']+|[\u0900-\u097F]+", re.UNICODE)


def normalize_transcript_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[-_/]+", " ", text)
    return " ".join(text.split())


def transcript_tokens(text: str) -> list[str]:
    return WORD_PATTERN.findall(normalize_transcript_text(text))


def token_language(token: str) -> str:
    has_devanagari = any("\u0900" <= char <= "\u097F" for char in token)
    has_latin = any(("a" <= char.lower() <= "z") for char in token)
    if has_devanagari and not has_latin:
        return "hindi"
    if has_latin and not has_devanagari:
        return "english"
    return "mixed"


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = transcript_tokens(reference)
    hyp_words = transcript_tokens(hypothesis)
    return word_error_rate_from_tokens(ref_words, hyp_words)


def word_error_rate_from_tokens(reference_tokens: list[str], hypothesis_tokens: list[str]) -> float:
    ref_words = reference_tokens
    hyp_words = hypothesis_tokens
    if not ref_words:
        return 0.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def frame_macro_f1(reference: torch.Tensor, prediction: torch.Tensor, num_classes: int = 2) -> float:
    scores = []
    for class_id in range(num_classes):
        ref_pos = reference == class_id
        pred_pos = prediction == class_id
        tp = (ref_pos & pred_pos).sum().item()
        fp = ((~ref_pos) & pred_pos).sum().item()
        fn = (ref_pos & (~pred_pos)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores))


def confusion_matrix_counts(reference: torch.Tensor, prediction: torch.Tensor, num_classes: int = 2) -> list[list[int]]:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for ref_class in range(num_classes):
        for pred_class in range(num_classes):
            matrix[ref_class, pred_class] = ((reference == ref_class) & (prediction == pred_class)).sum()
    return matrix.tolist()


def switch_accuracy_with_tolerance(
    reference_times: list[float],
    predicted_times: list[float],
    tolerance_sec: float = 0.2,
) -> float:
    if not reference_times:
        return 1.0
    matched = 0
    used = set()
    for ref in reference_times:
        for idx, pred in enumerate(predicted_times):
            if idx in used:
                continue
            if abs(ref - pred) <= tolerance_sec:
                matched += 1
                used.add(idx)
                break
    return matched / len(reference_times)


def switch_times_from_binary_sequence(sequence: torch.Tensor, hop_seconds: float) -> list[float]:
    indices = (sequence > 0).nonzero(as_tuple=False).flatten().tolist()
    return [idx * hop_seconds for idx in indices]


def equal_error_rate(target_scores: list[float], nontarget_scores: list[float]) -> float:
    _, eer = threshold_at_equal_error_rate(target_scores, nontarget_scores)
    return eer


def threshold_at_equal_error_rate(target_scores: list[float], nontarget_scores: list[float]) -> tuple[float, float]:
    thresholds = sorted(set(target_scores + nontarget_scores))
    if not thresholds:
        return 0.5, 1.0
    best_gap = float("inf")
    best_threshold = thresholds[0]
    best_eer = 1.0
    for threshold in thresholds:
        fa = sum(score >= threshold for score in nontarget_scores) / max(len(nontarget_scores), 1)
        fr = sum(score < threshold for score in target_scores) / max(len(target_scores), 1)
        gap = abs(fa - fr)
        if gap < best_gap:
            best_gap = gap
            best_threshold = threshold
            best_eer = (fa + fr) / 2.0
    return float(best_threshold), float(best_eer)


def cmvn(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(eps)
    return (features - mean) / std


def mel_cepstral_distortion(reference_mfcc: torch.Tensor, hypothesis_mfcc: torch.Tensor) -> float:
    if reference_mfcc.ndim != 2 or hypothesis_mfcc.ndim != 2:
        raise ValueError("MFCC tensors must be 2D with shape (frames, coefficients).")
    min_frames = min(reference_mfcc.size(0), hypothesis_mfcc.size(0))
    if min_frames == 0:
        return 0.0
    ref = cmvn(reference_mfcc[:min_frames])
    hyp = cmvn(hypothesis_mfcc[:min_frames])
    diff = ref[:, 1:] - hyp[:, 1:]
    constant = 6.141851463713754
    return float(constant * diff.pow(2).sum(dim=-1).sqrt().mean().item())
