from __future__ import annotations

import argparse
import copy
import csv
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

from assignment2.models.lid import FrameLevelLIDNet
from assignment2.modules.evaluation import cmvn
from assignment2.modules.features import log_mel_spectrogram
from assignment2.modules.lid_system import predict_lid_frames
from assignment2.modules.reporting import evaluate_lid_metrics, evaluate_transcription_metrics
from assignment2.utils.audio import load_audio
from pipeline import command_train_lid, command_transcribe, load_config


TASK_NAME = "Task 1 - Robust Code-Switched Transcription"
TASK_DIR = Path("outputs") / "task1"


def _select_device(requested: str) -> str:
    return requested if torch.cuda.is_available() and requested.startswith("cuda") else "cpu"


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _configure_task_paths(config: dict, stage: str) -> tuple[dict, Path]:
    task_config = copy.deepcopy(config)
    out_dir = TASK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = task_config["paths"]
    legacy_lid = Path(paths["lid_checkpoint"])
    task_lid = out_dir / "lid_frame_model.pt"
    use_task_lid = stage in {"all", "train-lid"} or task_lid.exists() or not legacy_lid.exists()

    paths["outputs_dir"] = str(out_dir)
    paths["lid_checkpoint"] = str(task_lid if use_task_lid else legacy_lid)
    paths["transcript_output"] = str(out_dir / "transcript_constrained.txt")
    paths["report_metrics_output"] = str(out_dir / "task1_metrics.json")
    return task_config, out_dir


def _copy_if_present(source: Path, target: Path) -> None:
    if source.exists() and source.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _organize_existing_artifacts(config: dict, out_dir: Path) -> None:
    _copy_if_present(Path(config["paths"]["lid_checkpoint"]), out_dir / "lid_frame_model.pt")
    _copy_if_present(Path("outputs") / "original_segment_denoised.wav", out_dir / "original_segment_denoised.wav")
    _copy_if_present(Path("outputs") / "transcript_constrained.txt", out_dir / "transcript_constrained.txt")
    _copy_if_present(Path("outputs") / "manual_transcript_excerpt.txt", out_dir / "manual_transcript_excerpt.txt")
    task_lid = out_dir / "lid_frame_model.pt"
    if task_lid.exists():
        config["paths"]["lid_checkpoint"] = str(task_lid)


def _save_matrix_plot(
    matrix: list[list[int]] | None,
    labels: list[str],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> Path | None:
    if not matrix:
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    max_value = max(max(row) for row in matrix) if matrix else 0
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            color = "white" if value > max_value / 2 else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_lid_confusion_matrix_plot(lid_metrics: dict[str, object], out_dir: Path) -> Path | None:
    if not lid_metrics.get("available"):
        return None

    title = "Task 1 LID Confusion Matrix"
    f1_score = lid_metrics.get("frame_macro_f1")
    if isinstance(f1_score, float):
        title = f"{title} (F1={f1_score:.4f})"
    return _save_matrix_plot(
        matrix=lid_metrics.get("confusion_matrix"),
        labels=["Hindi", "English"],
        title=title,
        x_label="Predicted language",
        y_label="Reference language",
        output_path=out_dir / "lid_confusion_matrix.png",
    )


def _save_switch_confusion_matrix_plot(lid_metrics: dict[str, object], out_dir: Path) -> Path | None:
    if not lid_metrics.get("available"):
        return None

    title = "Task 1 Code-Switch Boundary Confusion Matrix"
    switch_accuracy = lid_metrics.get("switch_accuracy_200ms")
    if isinstance(switch_accuracy, float):
        title = f"{title} (200ms acc={switch_accuracy:.4f})"
    return _save_matrix_plot(
        matrix=lid_metrics.get("switch_confusion_matrix"),
        labels=["Non-switch", "Switch"],
        title=title,
        x_label="Predicted boundary state",
        y_label="Reference boundary state",
        output_path=out_dir / "switch_confusion_matrix.png",
    )


def _save_wer_plot(transcription_metrics: dict[str, object], out_dir: Path) -> Path | None:
    if not transcription_metrics.get("available"):
        return None

    metric_order = [
        ("overall_wer", "Overall"),
        ("english_wer", "English"),
        ("hindi_wer", "Hindi"),
    ]
    points = [
        (label, float(transcription_metrics[key]))
        for key, label in metric_order
        if isinstance(transcription_metrics.get(key), float)
    ]
    if not points:
        return None

    import matplotlib.pyplot as plt

    labels = [label for label, _ in points]
    values = [value for _, value in points]
    thresholds = {
        "Overall": None,
        "English": 0.15,
        "Hindi": 0.25,
    }
    colors = []
    for label, value in points:
        threshold = thresholds.get(label)
        if threshold is None:
            colors.append("#4E79A7")
        else:
            colors.append("#59A14F" if value <= threshold else "#E15759")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, max(max(values) * 1.2, 0.3))
    ax.set_ylabel("Word Error Rate")
    ax.set_title("Task 1 Transcription Error Summary")
    ax.axhline(0.15, linestyle="--", linewidth=1.0, color="#59A14F", label="English target < 0.15")
    ax.axhline(0.25, linestyle=":", linewidth=1.0, color="#F28E2B", label="Hindi target < 0.25")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plot_path = out_dir / "wer_summary.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _predict_full_audio_lid_artifacts(config: dict, out_dir: Path, device: str) -> tuple[Path | None, Path | None]:
    checkpoint_path = Path(config["paths"]["lid_checkpoint"])
    original_audio_path = Path(config["paths"]["original_audio"])
    if not checkpoint_path.exists() or not original_audio_path.exists():
        return None, None

    audio_cfg = config["audio"]
    lid_cfg = config["lid"]
    sample_rate = int(audio_cfg["sample_rate"])
    hop_length = int(audio_cfg["hop_length"])
    hop_seconds = hop_length / sample_rate

    model = FrameLevelLIDNet(
        input_dim=lid_cfg["input_dim"],
        conv_dim=lid_cfg["conv_dim"],
        hidden_dim=lid_cfg["hidden_dim"],
        num_languages=lid_cfg["num_languages"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    waveform, _ = load_audio(original_audio_path, target_sr=sample_rate, mono=True)
    features = log_mel_spectrogram(
        waveform,
        sample_rate=sample_rate,
        n_fft=audio_cfg["n_fft"],
        win_length=audio_cfg["win_length"],
        hop_length=hop_length,
        n_mels=audio_cfg["n_mels"],
    ).squeeze(0)
    features = cmvn(features)

    chunk_frames = max(1, int(round(lid_cfg["chunk_seconds"] * sample_rate / hop_length)))
    language_predictions: list[torch.Tensor] = []

    for start in range(0, features.size(0), chunk_frames):
        end = min(start + chunk_frames, features.size(0))
        predictions = predict_lid_frames(
            model,
            features[start:end].to(device),
            english_enter_threshold=lid_cfg.get("english_enter_threshold", 0.55),
            english_exit_threshold=lid_cfg.get("english_exit_threshold", 0.18),
            smoothing_frames=lid_cfg.get("language_smoothing_frames", 11),
            minimum_english_frames=lid_cfg.get("minimum_english_frames", 5),
            switch_probability_threshold=lid_cfg.get("switch_probability_threshold", 0.35),
            switch_min_separation_frames=lid_cfg.get("switch_min_separation_frames", 6),
        )
        language_predictions.append(predictions["language_pred"].cpu())

    if not language_predictions:
        return None, None

    full_prediction = torch.cat(language_predictions)
    labels = ["hindi", "english"]
    segments: list[tuple[float, float, str]] = []
    segment_start = 0
    current_label = int(full_prediction[0].item())
    for frame_idx in range(1, full_prediction.numel()):
        next_label = int(full_prediction[frame_idx].item())
        if next_label != current_label:
            segments.append((segment_start * hop_seconds, frame_idx * hop_seconds, labels[current_label]))
            segment_start = frame_idx
            current_label = next_label
    segments.append((segment_start * hop_seconds, full_prediction.numel() * hop_seconds, labels[current_label]))

    csv_path = out_dir / "lid_full_audio_segments.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_sec", "end_sec", "pred_lang"])
        for start_sec, end_sec, label in segments:
            writer.writerow([f"{start_sec:.2f}", f"{end_sec:.2f}", label])

    import matplotlib.pyplot as plt

    plot_path = out_dir / "lid_full_audio_timeline.png"
    colors = {"hindi": "#4E79A7", "english": "#E15759"}
    fig, ax = plt.subplots(figsize=(12, 2.8))
    for start_sec, end_sec, label in segments:
        ax.axvspan(start_sec, end_sec, ymin=0.15, ymax=0.85, color=colors[label], alpha=0.9)
    ax.set_xlim(0, max(segments[-1][1], 1.0))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Task 1 Full-Audio Predicted LID Timeline")
    legend_handles = [
        plt.Line2D([0], [0], color=colors["hindi"], linewidth=10, label="Hindi"),
        plt.Line2D([0], [0], color=colors["english"], linewidth=10, label="English"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return csv_path, plot_path


def _write_summary(
    config: dict,
    out_dir: Path,
    device: str,
    reference_text_file: str | None,
    hypothesis_text_file: str | None,
) -> Path:
    reference_path = (
        Path(reference_text_file)
        if reference_text_file
        else _first_existing([out_dir / "manual_transcript_excerpt.txt", Path("outputs") / "manual_transcript_excerpt.txt"])
    )
    hypothesis_path = (
        Path(hypothesis_text_file)
        if hypothesis_text_file
        else _first_existing([out_dir / "transcript_constrained.txt", Path("outputs") / "transcript_constrained.txt"])
    )

    metrics = {
        "lid": evaluate_lid_metrics(config, device),
        "transcription": evaluate_transcription_metrics(
            reference_text_path=reference_path,
            hypothesis_text_path=hypothesis_path,
            segmented_eval_csv=config["paths"].get("transcript_eval_segments"),
            syllabus_text_path=config["paths"].get("syllabus_text"),
        ),
    }
    model_name = str(config["paths"].get("stt_model_name", ""))
    decoder_label = (
        "Syllabus-biased Whisper logit-bias decoding"
        if "whisper" in model_name.casefold()
        else "N-gram-biased constrained CTC beam search"
    )
    full_audio_lid_csv_path, full_audio_lid_plot_path = _predict_full_audio_lid_artifacts(config, out_dir, device)
    matrix_plot_path = _save_lid_confusion_matrix_plot(metrics["lid"], out_dir)
    switch_matrix_plot_path = _save_switch_confusion_matrix_plot(metrics["lid"], out_dir)
    wer_plot_path = _save_wer_plot(metrics["transcription"], out_dir)
    summary = {
        "task": TASK_NAME,
        "entrypoint": "python tasks/task1_robust_stt/main.py",
        "results_dir": str(out_dir),
        "core_code_files": [
            "src/assignment2/models/lid.py",
            "src/assignment2/modules/lid_system.py",
            "src/assignment2/modules/denoise.py",
            "src/assignment2/modules/ngram_lm.py",
            "src/assignment2/modules/ctc_decode.py",
            "src/assignment2/modules/stt_system.py",
        ],
        "outputs": {
            "lid_checkpoint": config["paths"]["lid_checkpoint"],
            "denoised_audio": str(out_dir / "original_segment_denoised.wav")
            if (out_dir / "original_segment_denoised.wav").exists()
            else None,
            "constrained_transcript": str(hypothesis_path) if hypothesis_path and hypothesis_path.exists() else None,
            "lid_confusion_matrix_plot": str(matrix_plot_path) if matrix_plot_path else None,
            "switch_confusion_matrix_plot": str(switch_matrix_plot_path) if switch_matrix_plot_path else None,
            "full_audio_lid_segments_csv": str(full_audio_lid_csv_path) if full_audio_lid_csv_path else None,
            "full_audio_lid_timeline_plot": str(full_audio_lid_plot_path) if full_audio_lid_plot_path else None,
            "wer_summary_plot": str(wer_plot_path) if wer_plot_path else None,
            "metrics": str(out_dir / "task1_metrics.json"),
        },
        "requirements_checked": {
            "task_1_1_lid_minimum_f1": 0.85,
            "task_1_2_constrained_decoding": decoder_label,
            "task_1_3_denoising": "Spectral subtraction followed by peak normalization",
        },
        "metrics": metrics,
    }

    summary_path = out_dir / "task1_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=TASK_NAME)
    parser.add_argument("--config", default="configs/assignment2_config.json")
    parser.add_argument(
        "--stage",
        choices=["all", "train-lid", "transcribe", "evaluate"],
        default="evaluate",
        help="Use 'all' for full reproduction. The default only writes/refreshes metrics from available artifacts.",
    )
    parser.add_argument("--reference-text-file", default=None)
    parser.add_argument("--hypothesis-text-file", default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    os.chdir(ROOT)
    args = build_argparser().parse_args()
    config, out_dir = _configure_task_paths(load_config(args.config), args.stage)
    device = _select_device(args.device)

    if args.stage in {"all", "train-lid"}:
        command_train_lid(config, device)
    if args.stage in {"all", "transcribe"}:
        command_transcribe(config, device)

    _organize_existing_artifacts(config, out_dir)
    summary_path = _write_summary(
        config=config,
        out_dir=out_dir,
        device=device,
        reference_text_file=args.reference_text_file,
        hypothesis_text_file=args.hypothesis_text_file,
    )
    print(f"Wrote {TASK_NAME} summary to {summary_path}")


if __name__ == "__main__":
    main()
