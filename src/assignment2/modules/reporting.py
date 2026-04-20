from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import torch

from assignment2.models.lid import FrameLevelLIDNet
from assignment2.modules.adversarial import search_fgsm_attack
from assignment2.modules.evaluation import (
    confusion_matrix_counts,
    frame_macro_f1,
    mel_cepstral_distortion,
    normalize_transcript_text,
    switch_accuracy_with_tolerance,
    switch_times_from_binary_sequence,
    token_language,
    transcript_tokens,
    word_error_rate,
    word_error_rate_from_tokens,
)
from assignment2.modules.features import mfcc_features
from assignment2.modules.lid_system import LIDSequenceDataset, predict_lid_frames
from assignment2.modules.ngram_lm import TERM_PATTERN
from assignment2.modules.spoof import build_spoof_segment_manifest, evaluate_spoof_experiment
from assignment2.utils.audio import load_audio, slice_audio


def _optional_path(path_like: str | Path | None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    return path if path.exists() else None


def _extract_syllabus_terms(syllabus_text: str) -> list[str]:
    normalized_lines = [normalize_transcript_text(line) for line in syllabus_text.splitlines() if line.strip()]
    terms = set(normalized_lines)
    terms.update(normalize_transcript_text(match.group(0)) for match in TERM_PATTERN.finditer(syllabus_text))
    return sorted((term for term in terms if term), key=len, reverse=True)


def _syllabus_term_metrics(reference: str, hypothesis: str, syllabus_text: str) -> dict[str, object]:
    reference_norm = normalize_transcript_text(reference)
    hypothesis_norm = normalize_transcript_text(hypothesis)
    terms = _extract_syllabus_terms(syllabus_text)
    reference_hits = sorted({term for term in terms if term in reference_norm})
    hypothesis_hits = sorted({term for term in terms if term in hypothesis_norm})
    shared_hits = sorted(set(reference_hits) & set(hypothesis_hits))

    result: dict[str, object] = {
        "syllabus_terms_total": len(terms),
        "reference_syllabus_term_count": len(reference_hits),
        "hypothesis_syllabus_term_count": len(hypothesis_hits),
        "shared_syllabus_term_count": len(shared_hits),
        "shared_syllabus_terms": shared_hits[:25],
    }
    if reference_hits:
        result["shared_syllabus_term_recall_vs_reference"] = len(shared_hits) / len(reference_hits)
    else:
        result["shared_syllabus_term_recall_vs_reference"] = None
        result["syllabus_term_note"] = (
            "The provided reference transcript contains no terms from data/syllabus_text.txt, "
            "so technical-term prioritization cannot be quantified on this audio."
        )
    return result


def evaluate_transcription_metrics(
    reference_text_path: str | Path | None,
    hypothesis_text_path: str | Path | None,
    segmented_eval_csv: str | Path | None = None,
    syllabus_text_path: str | Path | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {"available": False}
    reference_path = _optional_path(reference_text_path)
    hypothesis_path = _optional_path(hypothesis_text_path)
    if reference_path is None or hypothesis_path is None:
        result["reason"] = "Reference or hypothesis transcript file is missing."
        return result

    reference = reference_path.read_text(encoding="utf-8")
    hypothesis = hypothesis_path.read_text(encoding="utf-8")
    result.update(
        {
            "available": True,
            "reference_path": str(reference_path),
            "hypothesis_path": str(hypothesis_path),
            "reference_word_count": len(transcript_tokens(reference)),
            "hypothesis_word_count": len(transcript_tokens(hypothesis)),
            "overall_wer": float(word_error_rate(reference, hypothesis)),
        }
    )

    syllabus_path = _optional_path(syllabus_text_path)
    if syllabus_path is not None:
        syllabus_text = syllabus_path.read_text(encoding="utf-8")
        result.update(_syllabus_term_metrics(reference, hypothesis, syllabus_text))

    segmented_rows = 0
    segmented_path = _optional_path(segmented_eval_csv)
    if segmented_path is not None:
        per_language: dict[str, list[float]] = {}
        with segmented_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row["reference_text"].strip() or not row["hypothesis_text"].strip():
                    continue
                lang = row["lang"].strip().lower()
                per_language.setdefault(lang, []).append(word_error_rate(row["reference_text"], row["hypothesis_text"]))
                segmented_rows += 1
        for lang, values in per_language.items():
            result[f"{lang}_wer"] = float(sum(values) / max(len(values), 1))
        if segmented_rows > 0:
            result["per_language_wer_method"] = "segment_csv"

    if segmented_rows == 0:
        reference_tokens = transcript_tokens(reference)
        hypothesis_tokens = transcript_tokens(hypothesis)
        for lang in ("english", "hindi"):
            ref_subset = [token for token in reference_tokens if token_language(token) == lang]
            hyp_subset = [token for token in hypothesis_tokens if token_language(token) == lang]
            if ref_subset:
                result[f"{lang}_wer"] = float(word_error_rate_from_tokens(ref_subset, hyp_subset))
        if any(key in result for key in ("english_wer", "hindi_wer")):
            result["per_language_wer_method"] = "script_filtered_full_transcript"
    return result


def evaluate_lid_metrics(config: dict, device: str) -> dict[str, object]:
    checkpoint_path = _optional_path(config["paths"].get("lid_checkpoint"))
    manifest_path = _optional_path(config["paths"].get("lid_manifest"))
    if checkpoint_path is None or manifest_path is None:
        return {
            "available": False,
            "reason": "LID checkpoint or manifest is missing.",
        }

    audio_cfg = config["audio"]
    lid_cfg = config["lid"]
    dataset = LIDSequenceDataset(
        manifest_path=manifest_path,
        sample_rate=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        win_length=audio_cfg["win_length"],
        hop_length=audio_cfg["hop_length"],
        n_mels=audio_cfg["n_mels"],
        chunk_seconds=lid_cfg["chunk_seconds"],
        switch_boundary_radius_frames=lid_cfg.get("switch_boundary_radius_frames", 6),
    )
    if len(dataset) == 0:
        return {
            "available": False,
            "reason": "LID manifest produced no labeled chunks.",
        }

    model = FrameLevelLIDNet(
        input_dim=lid_cfg["input_dim"],
        conv_dim=lid_cfg["conv_dim"],
        hidden_dim=lid_cfg["hidden_dim"],
        num_languages=lid_cfg["num_languages"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    reference_frames: list[torch.Tensor] = []
    predicted_frames: list[torch.Tensor] = []
    reference_switch_frames: list[torch.Tensor] = []
    predicted_switch_frames: list[torch.Tensor] = []
    reference_switch_times: list[float] = []
    predicted_switch_times: list[float] = []
    hop_seconds = audio_cfg["hop_length"] / audio_cfg["sample_rate"]

    for example in dataset:
        predictions = predict_lid_frames(
            model,
            example.features.to(device),
            english_enter_threshold=lid_cfg.get("english_enter_threshold", 0.55),
            english_exit_threshold=lid_cfg.get("english_exit_threshold", 0.18),
            smoothing_frames=lid_cfg.get("language_smoothing_frames", 11),
            minimum_english_frames=lid_cfg.get("minimum_english_frames", 5),
            switch_probability_threshold=lid_cfg.get("switch_probability_threshold", 0.35),
            switch_min_separation_frames=lid_cfg.get("switch_min_separation_frames", 6),
        )
        valid = example.language_targets >= 0
        if valid.any():
            reference_frames.append(example.language_targets[valid].cpu())
            predicted_frames.append(predictions["language_pred"][valid].cpu())
            reference_switch_frames.append(example.switch_eval_targets[valid].cpu())
            predicted_switch_frames.append(predictions["switch_pred"][valid].cpu())
        reference_switch_times.extend(switch_times_from_binary_sequence(example.switch_eval_targets.cpu(), hop_seconds))
        predicted_binary = predictions["switch_pred"].cpu()
        predicted_switch_times.extend(switch_times_from_binary_sequence(predicted_binary, hop_seconds))

    if not reference_frames:
        return {
            "available": False,
            "reason": "No valid frame labels were found in the LID manifest.",
        }

    reference = torch.cat(reference_frames)
    prediction = torch.cat(predicted_frames)
    reference_switch = torch.cat(reference_switch_frames)
    predicted_switch = torch.cat(predicted_switch_frames)
    labeled_seconds = float(reference.numel() * hop_seconds)
    original_audio_path = _optional_path(config["paths"].get("original_audio"))
    full_audio_seconds = None
    if original_audio_path is not None and original_audio_path.suffix.lower() == ".wav":
        with wave.open(str(original_audio_path), "rb") as wav_file:
            full_audio_seconds = wav_file.getnframes() / float(wav_file.getframerate())

    coverage_ratio = None
    if full_audio_seconds and full_audio_seconds > 0:
        coverage_ratio = labeled_seconds / full_audio_seconds

    return {
        "available": True,
        "frame_macro_f1": float(frame_macro_f1(reference, prediction, num_classes=lid_cfg["num_languages"])),
        "switch_accuracy_200ms": float(
            switch_accuracy_with_tolerance(reference_switch_times, predicted_switch_times, tolerance_sec=0.2)
        ),
        "confusion_matrix": confusion_matrix_counts(reference, prediction, num_classes=lid_cfg["num_languages"]),
        "switch_confusion_matrix": confusion_matrix_counts(reference_switch, predicted_switch, num_classes=2),
        "num_switch_frames": int(reference_switch.sum().item()),
        "num_labeled_frames": int(reference.numel()),
        "labeled_audio_seconds": labeled_seconds,
        "full_audio_seconds": full_audio_seconds,
        "labeled_audio_coverage_ratio": coverage_ratio,
    }


def evaluate_tts_metrics(config: dict) -> dict[str, object]:
    reference_path = _optional_path(config["paths"].get("student_voice_audio"))
    hypothesis_path = _optional_path(Path(config["paths"]["outputs_dir"]) / "output_LRL_cloned.wav")
    if reference_path is None or hypothesis_path is None:
        return {
            "available": False,
            "reason": "Reference voice or synthesized spoof audio is missing.",
        }

    prefix_seconds = float(config.get("reporting", {}).get("mcd_prefix_seconds", 30.0))
    sample_rate = int(config["audio"]["sample_rate"])
    reference_waveform, _ = load_audio(reference_path, target_sr=sample_rate, mono=True)
    hypothesis_waveform, _ = load_audio(hypothesis_path, target_sr=sample_rate, mono=True)
    prefix = min(prefix_seconds, reference_waveform.size(-1) / sample_rate, hypothesis_waveform.size(-1) / sample_rate)
    reference_slice = slice_audio(reference_waveform, sample_rate=sample_rate, start_sec=0.0, end_sec=prefix)
    hypothesis_slice = slice_audio(hypothesis_waveform, sample_rate=sample_rate, start_sec=0.0, end_sec=prefix)
    reference_mfcc = mfcc_features(reference_slice, sample_rate=sample_rate, n_mfcc=13).squeeze(0)
    hypothesis_mfcc = mfcc_features(hypothesis_slice, sample_rate=sample_rate, n_mfcc=13).squeeze(0)
    return {
        "available": True,
        "approx_mcd_prefix_seconds": float(prefix),
        "approx_mcd_cmvn": float(mel_cepstral_distortion(reference_mfcc, hypothesis_mfcc)),
        "note": "This is a prefix-aligned proxy because the 60-second reference and synthesized lecture are not parallel same-text utterances.",
    }


def evaluate_attack_metrics(config: dict, device: str) -> dict[str, object]:
    checkpoint_path = _optional_path(config["paths"].get("lid_checkpoint"))
    original_audio = _optional_path(config["paths"].get("original_audio"))
    if checkpoint_path is None or original_audio is None:
        return {
            "available": False,
            "reason": "LID checkpoint or original audio is missing.",
        }

    audio_cfg = config["audio"]
    lid_cfg = config["lid"]
    model = FrameLevelLIDNet(
        input_dim=lid_cfg["input_dim"],
        conv_dim=lid_cfg["conv_dim"],
        hidden_dim=lid_cfg["hidden_dim"],
        num_languages=lid_cfg["num_languages"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    waveform, sample_rate = load_audio(original_audio, target_sr=audio_cfg["sample_rate"], mono=True)
    search_result = search_fgsm_attack(
        model=model,
        waveform=waveform.to(device),
        sample_rate=sample_rate,
        source_label=config["adversarial"].get("source_label", 0),
        target_label=config["adversarial"]["target_label"],
        epsilon_grid=config["adversarial"]["epsilon_grid"],
        snr_threshold_db=config["adversarial"]["snr_threshold_db"],
        n_fft=audio_cfg["n_fft"],
        win_length=audio_cfg["win_length"],
        hop_length=audio_cfg["hop_length"],
        n_mels=audio_cfg["n_mels"],
        segment_seconds=config["adversarial"].get("segment_seconds", 5.0),
        stride_seconds=config["adversarial"].get("stride_seconds", 2.5),
        max_candidate_segments=config["adversarial"].get("max_candidate_segments", 8),
        attack_steps=config["adversarial"].get("attack_steps", 10),
        success_ratio_threshold=config["adversarial"].get("success_ratio_threshold", 0.5),
    )
    if search_result is None:
        return {
            "available": False,
            "reason": "No valid 5-second attack segments were found.",
        }
    successful_trial = next((trial for trial in search_result.trials if trial.success), None)
    return {
        "available": successful_trial is not None,
        "minimum_epsilon": float(successful_trial.epsilon) if successful_trial is not None else None,
        "snr_threshold_db": float(config["adversarial"]["snr_threshold_db"]),
        "segment_start_sec": float(search_result.segment_start_sec),
        "segment_end_sec": float(search_result.segment_end_sec),
        "baseline_source_ratio": float(search_result.baseline_source_ratio),
        "epsilon_trials": [
            {
                "epsilon": float(trial.epsilon),
                "snr_db": float(trial.snr_db),
                "target_frame_ratio": float(trial.target_frame_ratio),
                "success": bool(trial.success),
            }
            for trial in search_result.trials
        ],
    }


def run_report_metrics(
    config: dict,
    device: str,
    reference_text_path: str | Path | None = None,
    hypothesis_text_path: str | Path | None = None,
) -> dict[str, object]:
    outputs_dir = Path(config["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reporting_cfg = config.get("reporting", {})

    spoof_audio_path = _optional_path(config["paths"].get("spoof_audio"))
    if spoof_audio_path is None:
        spoof_audio_path = _optional_path(outputs_dir / "output_LRL_cloned.wav")

    if spoof_audio_path is not None:
        spoof_manifest_path = Path(config["paths"].get("spoof_eval_manifest", "data/manifests/spoof_segment_eval.csv"))
        spoof_summary = build_spoof_segment_manifest(
            bona_fide_audio=config["paths"]["student_voice_audio"],
            spoof_audio=spoof_audio_path,
            output_manifest=spoof_manifest_path,
            sample_rate=config["audio"]["sample_rate"],
            chunk_seconds=float(reporting_cfg.get("spoof_chunk_seconds", 4.0)),
            group_chunks=int(reporting_cfg.get("spoof_group_chunks", 2)),
            max_chunks_per_label=int(reporting_cfg.get("spoof_max_chunks_per_label", 60)),
        )
        spoof_metrics: dict[str, object] = evaluate_spoof_experiment(
            manifest_path=spoof_manifest_path,
            checkpoint_path=Path(config["paths"].get("spoof_checkpoint", Path("models") / "anti_spoof_lfcc.pt")),
            sample_rate=config["audio"]["sample_rate"],
            n_lfcc=config["spoof"]["n_lfcc"],
            batch_size=config["spoof"]["batch_size"],
            epochs=config["spoof"]["epochs"],
            learning_rate=config["spoof"]["learning_rate"],
            device=device,
            roc_plot_path=outputs_dir / "anti_spoof_roc.png",
            confusion_plot_path=outputs_dir / "anti_spoof_confusion_matrix.png",
        )
        spoof_result: dict[str, object] = {**spoof_summary, **spoof_metrics}
    else:
        spoof_result = {
            "available": False,
            "reason": "Synthesized spoof audio is missing.",
        }

    results = {
        "transcription": evaluate_transcription_metrics(
            reference_text_path=reference_text_path,
            hypothesis_text_path=hypothesis_text_path,
            segmented_eval_csv=config["paths"].get("transcript_eval_segments"),
            syllabus_text_path=config["paths"].get("syllabus_text"),
        ),
        "lid": evaluate_lid_metrics(config, device),
        "tts": evaluate_tts_metrics(config),
        "spoof": spoof_result,
        "adversarial": evaluate_attack_metrics(config, device),
    }

    metrics_path = Path(config["paths"].get("report_metrics_output", outputs_dir / "report_metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (outputs_dir / "anti_spoof_metrics.json").write_text(json.dumps(results["spoof"], indent=2), encoding="utf-8")
    return results
