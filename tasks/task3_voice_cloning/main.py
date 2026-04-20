from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assignment2.modules.features import normalize_contour
from assignment2.modules.prosody import extract_prosody
from assignment2.modules.reporting import evaluate_tts_metrics
from assignment2.utils.audio import frame_rms, load_audio, rms_normalize, save_audio
from pipeline import command_speaker_embed, command_synthesize, load_config


TASK_NAME = "Task 3 - Cross-Lingual Voice Cloning"
TASK_DIR = Path("outputs") / "task3"


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _configure_task_paths(config: dict, text_file: str | None) -> tuple[dict, Path, Path | None]:
    task_config = copy.deepcopy(config)
    out_dir = TASK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_text = Path(text_file) if text_file else _first_existing(
        [
            Path("outputs") / "task2" / "translation_gujarati_tts.txt",
            Path("outputs") / "task2" / "translation_gujarati.txt",
            Path("outputs") / "translation_gujarati.txt",
        ]
    )
    task_config["paths"]["outputs_dir"] = str(out_dir)
    if selected_text is not None:
        task_config["paths"]["translation_output"] = str(selected_text)
    return task_config, out_dir, selected_text


def _copy_if_present(source: Path, target: Path, overwrite: bool = True) -> None:
    if source.exists() and source.resolve() != target.resolve():
        if target.exists() and not overwrite:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _organize_existing_artifacts(out_dir: Path) -> None:
    _copy_if_present(Path("outputs") / "student_voice_embedding.pt", out_dir / "student_voice_embedding.pt", overwrite=False)
    _copy_if_present(Path("outputs") / "output_LRL_cloned.wav", out_dir / "output_LRL_cloned.wav", overwrite=False)
    _copy_if_present(Path("outputs") / "output_LRL_flat.wav", out_dir / "output_LRL_flat.wav", overwrite=False)
    _copy_if_present(Path("outputs") / "prosody_diagnostic.png", out_dir / "prosody_diagnostic.png", overwrite=False)
    _copy_if_present(Path("outputs") / "original_vs_spoof_prosody.png", out_dir / "original_vs_spoof_prosody.png", overwrite=False)


def _prepare_exact_reference_audio(config: dict, out_dir: Path, target_seconds: float = 60.0) -> Path | None:
    source_path = Path(config["paths"]["student_voice_audio"])
    if not source_path.exists():
        return None

    waveform, sample_rate = load_audio(source_path, mono=True)
    target_samples = int(round(target_seconds * sample_rate))
    if waveform.size(-1) >= target_samples:
        adjusted = waveform[..., :target_samples]
    else:
        padding = torch.zeros((waveform.size(0), target_samples - waveform.size(-1)), dtype=waveform.dtype)
        adjusted = torch.cat([waveform, padding], dim=-1)

    exact_path = out_dir / "student_voice_exact_60s.wav"
    save_audio(exact_path, adjusted, sample_rate)
    config["paths"]["student_voice_audio"] = str(exact_path)
    return exact_path


def _backup_existing_task_audio(out_dir: Path) -> None:
    final_path = out_dir / "output_LRL_cloned.wav"
    if final_path.exists():
        _copy_if_present(final_path, out_dir / "output_LRL_cloned_previous.wav")


def _limit_peak(waveform: torch.Tensor, peak: float = 0.95, eps: float = 1e-8) -> torch.Tensor:
    current_peak = waveform.abs().amax().clamp_min(eps)
    if float(current_peak) <= peak:
        return waveform
    return waveform * (peak / current_peak)


def _soft_clip(waveform: torch.Tensor, drive: float = 2.5) -> torch.Tensor:
    drive_value = torch.tensor(drive, dtype=waveform.dtype, device=waveform.device)
    return torch.tanh(waveform * drive_value) / torch.tanh(drive_value)


def _band_limit_speech(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if waveform.numel() == 0:
        return waveform
    spectrum = torch.fft.rfft(waveform, dim=-1)
    frequencies = torch.fft.rfftfreq(waveform.size(-1), d=1.0 / sample_rate).to(waveform.device)
    high_cut = max(1000.0, min(7600.0, (sample_rate / 2.0) - 250.0))
    highpass = torch.clamp((frequencies - 65.0) / 55.0, min=0.0, max=1.0)
    lowpass = torch.clamp((high_cut - frequencies) / 900.0, min=0.0, max=1.0)
    mask = (highpass * lowpass).view(1, -1)
    return torch.fft.irfft(spectrum * mask, n=waveform.size(-1), dim=-1)


def _spectral_gate(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    n_fft = 1024 if sample_rate >= 16000 else 512
    win_length = n_fft
    hop_length = n_fft // 4
    window = torch.hann_window(win_length, device=waveform.device)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    magnitude = spec.abs()
    phase = torch.angle(spec)
    noise_floor = torch.quantile(magnitude, 0.20, dim=-1, keepdim=True)
    threshold = noise_floor * 1.65
    soft_mask = torch.clamp((magnitude - threshold) / (magnitude + 1e-8), min=0.10, max=1.0)
    clean_spec = torch.polar(magnitude * soft_mask, phase)
    clean = torch.istft(
        clean_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=waveform.size(-1),
    )
    return clean.unsqueeze(0) if clean.dim() == 1 else clean


def _audio_quality_stats(waveform: torch.Tensor, sample_rate: int) -> dict[str, float]:
    frame_length = min(max(int(0.025 * sample_rate), 1), waveform.size(-1))
    hop_length = min(max(int(0.010 * sample_rate), 1), frame_length)
    rms_values = frame_rms(waveform, frame_length=frame_length, hop_length=hop_length)
    noise_floor = torch.quantile(rms_values, 0.10).clamp_min(1e-8)
    full_rms = waveform.pow(2).mean().sqrt().clamp_min(1e-8)
    peak = waveform.abs().amax().clamp_min(1e-8)
    return {
        "duration_sec": float(waveform.size(-1) / sample_rate),
        "rms_dbfs": float(20.0 * torch.log10(full_rms)),
        "peak_dbfs": float(20.0 * torch.log10(peak)),
        "noise_floor_proxy_dbfs": float(20.0 * torch.log10(noise_floor)),
        "speech_to_noise_proxy_db": float(20.0 * torch.log10(full_rms / noise_floor)),
    }


def _enhance_cloned_audio(out_dir: Path, refresh_raw: bool = False) -> tuple[Path | None, Path | None]:
    final_path = out_dir / "output_LRL_cloned.wav"
    raw_path = out_dir / "output_LRL_cloned_raw.wav"
    clean_path = out_dir / "output_LRL_cloned_clean.wav"
    report_path = out_dir / "audio_cleanup_report.json"
    if not final_path.exists() and not raw_path.exists():
        return None, None

    if final_path.exists() and (refresh_raw or not raw_path.exists()):
        shutil.copy2(final_path, raw_path)

    waveform, sample_rate = load_audio(raw_path, mono=True)
    original_stats = _audio_quality_stats(waveform, sample_rate)
    normalized_input = rms_normalize(waveform, target_rms=0.055)
    normalized_input = _soft_clip(normalized_input, drive=2.5)
    normalized_input = _limit_peak(normalized_input, peak=0.95)

    enhanced = _band_limit_speech(normalized_input, sample_rate)
    enhanced = _spectral_gate(enhanced, sample_rate)
    enhanced = rms_normalize(enhanced, target_rms=0.055)
    enhanced = _soft_clip(enhanced, drive=1.6)
    enhanced = _limit_peak(enhanced, peak=0.95)

    enhanced_stats = _audio_quality_stats(enhanced, sample_rate)
    cleanup_report = {
        "input_raw_audio": str(raw_path),
        "clean_audio": str(clean_path),
        "final_audio_for_evaluation": str(final_path),
        "method": "FFT speech band-limit + quantile spectral gate + RMS and peak normalization",
        "original": original_stats,
        "normalized_input": _audio_quality_stats(normalized_input, sample_rate),
        "enhanced": enhanced_stats,
        "noise_floor_reduction_db": float(
            original_stats["noise_floor_proxy_dbfs"] - enhanced_stats["noise_floor_proxy_dbfs"]
        ),
        "speech_to_noise_proxy_improvement_db": float(
            enhanced_stats["speech_to_noise_proxy_db"] - original_stats["speech_to_noise_proxy_db"]
        ),
    }
    save_audio(clean_path, enhanced, sample_rate)
    save_audio(final_path, enhanced, sample_rate)
    report_path.write_text(json.dumps(cleanup_report, indent=2), encoding="utf-8")
    return clean_path, report_path


def _contour_mae(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    frames = min(reference.numel(), candidate.numel())
    if frames == 0:
        return 0.0
    return float((reference[:frames] - candidate[:frames]).abs().mean().item())


def _save_prosody_plots(config: dict, out_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    original_audio = Path(config["paths"]["original_audio"])
    final_audio = out_dir / "output_LRL_cloned.wav"
    flat_audio = out_dir / "output_LRL_flat.wav"
    if not original_audio.exists() or not final_audio.exists():
        return None, None, None

    import matplotlib.pyplot as plt

    sample_rate = int(config["tts"]["sample_rate"])
    frame_length = int(config["audio"]["win_length"])
    hop_length = int(config["audio"]["hop_length"])
    hop_ms = float(config["audio"]["hop_ms"])
    compare_seconds = float(config.get("reporting", {}).get("mcd_prefix_seconds", 30.0))

    professor_waveform, _ = load_audio(original_audio, target_sr=sample_rate, mono=True)
    warped_waveform, _ = load_audio(final_audio, target_sr=sample_rate, mono=True)
    compare_samples = min(
        professor_waveform.size(-1),
        warped_waveform.size(-1),
        int(compare_seconds * sample_rate),
    )
    if compare_samples <= 0:
        return None, None, None

    professor_profile = extract_prosody(
        professor_waveform[..., :compare_samples],
        sample_rate=sample_rate,
        hop_ms=hop_ms,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    warped_profile = extract_prosody(
        warped_waveform[..., :compare_samples],
        sample_rate=sample_rate,
        hop_ms=hop_ms,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    frame_count = min(professor_profile.pitch.numel(), warped_profile.pitch.numel())
    if frame_count == 0:
        return None, None, None

    time_axis = torch.arange(frame_count, dtype=torch.float32) * (hop_length / sample_rate)
    professor_pitch = professor_profile.pitch[:frame_count]
    warped_pitch = warped_profile.pitch[:frame_count]
    professor_energy = professor_profile.energy[:frame_count]
    warped_energy = warped_profile.energy[:frame_count]

    diagnostic_path = out_dir / "prosody_diagnostic.png"
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(time_axis.tolist(), professor_pitch.tolist(), label="Professor", linewidth=1.3)
    axes[0].plot(time_axis.tolist(), warped_pitch.tolist(), label="Warped TTS", linewidth=1.1)
    axes[0].set_ylabel("Pitch (Hz)")
    axes[0].set_title("Task 3 Prosody Diagnostic")
    axes[0].legend(loc="upper right")
    axes[1].plot(time_axis.tolist(), professor_energy.tolist(), label="Professor", linewidth=1.3)
    axes[1].plot(time_axis.tolist(), warped_energy.tolist(), label="Warped TTS", linewidth=1.1)
    axes[1].set_ylabel("Energy (RMS)")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(diagnostic_path, dpi=180)
    plt.close(fig)

    overlay_path = out_dir / "original_vs_spoof_prosody.png"
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(time_axis.tolist(), normalize_contour(professor_pitch).tolist(), label="Professor", linewidth=1.3)
    axes[0].plot(time_axis.tolist(), normalize_contour(warped_pitch).tolist(), label="Warped TTS", linewidth=1.1)
    axes[0].set_ylabel("Normalized pitch")
    axes[0].set_title("Normalized Prosody Alignment")
    axes[0].legend(loc="upper right")
    axes[1].plot(time_axis.tolist(), normalize_contour(professor_energy).tolist(), label="Professor", linewidth=1.3)
    axes[1].plot(time_axis.tolist(), normalize_contour(warped_energy).tolist(), label="Warped TTS", linewidth=1.1)
    axes[1].set_ylabel("Normalized energy")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=180)
    plt.close(fig)

    ablation_path: Path | None = None
    if flat_audio.exists():
        flat_waveform, _ = load_audio(flat_audio, target_sr=sample_rate, mono=True)
        flat_compare_samples = min(flat_waveform.size(-1), compare_samples)
        if flat_compare_samples > 0:
            flat_profile = extract_prosody(
                flat_waveform[..., :flat_compare_samples],
                sample_rate=sample_rate,
                hop_ms=hop_ms,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            flat_frames = min(frame_count, flat_profile.pitch.numel(), flat_profile.energy.numel())
            if flat_frames > 0:
                ablation_path = out_dir / "prosody_ablation.png"
                pitch_errors = [
                    _contour_mae(professor_pitch[:flat_frames], flat_profile.pitch[:flat_frames]),
                    _contour_mae(professor_pitch[:flat_frames], warped_pitch[:flat_frames]),
                ]
                energy_errors = [
                    _contour_mae(professor_energy[:flat_frames], flat_profile.energy[:flat_frames]),
                    _contour_mae(professor_energy[:flat_frames], warped_energy[:flat_frames]),
                ]
                labels = ["Flat TTS", "Warped TTS"]
                positions = [0, 1]
                width = 0.35
                ymax = max(pitch_errors + energy_errors + [1e-6])
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                ax.bar([pos - (width / 2.0) for pos in positions], pitch_errors, width=width, label="Pitch MAE")
                ax.bar([pos + (width / 2.0) for pos in positions], energy_errors, width=width, label="Energy MAE")
                ax.set_xticks(positions, labels=labels)
                ax.set_ylabel("Alignment error")
                ax.set_title("Prosody Warping Ablation")
                ax.legend(loc="upper right")
                for pos, value in zip([pos - (width / 2.0) for pos in positions], pitch_errors):
                    ax.text(pos, value + ymax * 0.03, f"{value:.3f}", ha="center")
                for pos, value in zip([pos + (width / 2.0) for pos in positions], energy_errors):
                    ax.text(pos, value + ymax * 0.03, f"{value:.3f}", ha="center")
                fig.tight_layout()
                fig.savefig(ablation_path, dpi=180)
                plt.close(fig)

    return diagnostic_path, overlay_path, ablation_path


def _save_tts_metric_plot(metrics: dict[str, object], out_dir: Path) -> Path | None:
    if not metrics.get("available"):
        return None

    approx_mcd = metrics.get("approx_mcd_cmvn")
    if not isinstance(approx_mcd, float):
        return None

    import matplotlib.pyplot as plt

    plot_path = out_dir / "tts_metric_summary.png"
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["Approx. MCD", "Target"], [approx_mcd, 8.0], color=["#4E79A7", "#59A14F"])
    ax.set_ylabel("MCD")
    ax.set_title("Task 3 Voice Similarity Proxy")
    ax.axhline(8.0, linestyle="--", linewidth=1.0, color="#59A14F", label="Assignment target < 8.0")
    for bar, value in zip(bars, [approx_mcd, 8.0]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.08, f"{value:.3f}", ha="center")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _write_summary(config: dict, out_dir: Path, selected_text: Path | None, exact_reference_path: Path | None) -> Path:
    metrics = evaluate_tts_metrics(config)
    cleanup_report_path = out_dir / "audio_cleanup_report.json"
    summary_path = out_dir / "task3_metrics.json"
    diagnostic_plot_path, overlay_plot_path, ablation_plot_path = _save_prosody_plots(config, out_dir)
    tts_metric_plot_path = _save_tts_metric_plot(metrics, out_dir)
    summary = {
        "task": TASK_NAME,
        "entrypoint": "python tasks/task3_voice_cloning/main.py",
        "results_dir": str(out_dir),
        "input_translation": str(selected_text) if selected_text else None,
        "core_code_files": [
            "src/assignment2/modules/speaker.py",
            "src/assignment2/modules/prosody.py",
            "src/assignment2/modules/tts.py",
        ],
        "outputs": {
            "student_voice_exact_60s": str(exact_reference_path) if exact_reference_path else None,
            "student_voice_embedding": str(out_dir / "student_voice_embedding.pt"),
            "flat_lrl_audio": str(out_dir / "output_LRL_flat.wav"),
            "cloned_lrl_audio": str(out_dir / "output_LRL_cloned.wav"),
            "raw_cloned_lrl_audio": str(out_dir / "output_LRL_cloned_raw.wav"),
            "clean_cloned_lrl_audio": str(out_dir / "output_LRL_cloned_clean.wav"),
            "prosody_diagnostic_plot": str(diagnostic_plot_path) if diagnostic_plot_path else None,
            "normalized_prosody_plot": str(overlay_plot_path) if overlay_plot_path else None,
            "prosody_ablation_plot": str(ablation_plot_path) if ablation_plot_path else None,
            "tts_metric_plot": str(tts_metric_plot_path) if tts_metric_plot_path else None,
            "audio_cleanup_report": str(cleanup_report_path),
            "metrics": str(summary_path),
        },
        "metrics": {
            "tts": metrics,
            "audio_cleanup": json.loads(cleanup_report_path.read_text(encoding="utf-8"))
            if cleanup_report_path.exists()
            else {"available": False, "reason": "No cloned audio was available for cleanup."},
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=TASK_NAME)
    parser.add_argument("--config", default="configs/assignment2_config.json")
    parser.add_argument(
        "--stage",
        choices=["all", "speaker-embed", "synthesize", "evaluate"],
        default="evaluate",
        help="Use 'all' to extract the embedding, synthesize audio, and write metrics.",
    )
    parser.add_argument("--text-file", default=None)
    return parser


def main() -> None:
    os.chdir(ROOT)
    args = build_argparser().parse_args()
    config, out_dir, selected_text = _configure_task_paths(load_config(args.config), args.text_file)
    _organize_existing_artifacts(out_dir)
    exact_reference_path = _prepare_exact_reference_audio(config, out_dir)

    if args.stage in {"all", "speaker-embed"}:
        command_speaker_embed(config)
    if args.stage in {"all", "synthesize"}:
        if selected_text is None:
            raise FileNotFoundError("No Gujarati translation file was found for Task 3 synthesis.")
        _backup_existing_task_audio(out_dir)
        command_synthesize(config, str(selected_text))

    _enhance_cloned_audio(out_dir, refresh_raw=args.stage in {"all", "synthesize"})
    summary_path = _write_summary(config, out_dir, selected_text, exact_reference_path)
    print(f"Wrote {TASK_NAME} summary to {summary_path}")


if __name__ == "__main__":
    main()
