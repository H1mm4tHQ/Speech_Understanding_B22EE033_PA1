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

from assignment2.modules.reporting import evaluate_attack_metrics
from assignment2.modules.spoof import build_spoof_segment_manifest, evaluate_spoof_experiment
from pipeline import command_attack, load_config


TASK_NAME = "Task 4 - Robustness and Spoof Detection"
TASK_DIR = Path("outputs") / "task4"


def _select_device(requested: str) -> str:
    return requested if torch.cuda.is_available() and requested.startswith("cuda") else "cpu"


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _configure_task_paths(config: dict, spoof_audio: str | None) -> tuple[dict, Path, Path | None]:
    task_config = copy.deepcopy(config)
    out_dir = TASK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_spoof = Path(spoof_audio) if spoof_audio else _first_existing(
        [
            Path("outputs") / "task3" / "output_LRL_cloned_clean.wav",
            Path("outputs") / "task3" / "output_LRL_cloned.wav",
            Path("outputs") / "output_LRL_cloned.wav",
        ]
    )
    selected_lid = _first_existing(
        [
            Path("outputs") / "task1" / "lid_frame_model.pt",
            Path(task_config["paths"]["lid_checkpoint"]),
        ]
    )

    paths = task_config["paths"]
    paths["outputs_dir"] = str(out_dir)
    paths["spoof_eval_manifest"] = str(out_dir / "spoof_segment_eval.csv")
    paths["spoof_checkpoint"] = str(out_dir / "anti_spoof_lfcc.pt")
    paths["report_metrics_output"] = str(out_dir / "task4_metrics.json")
    if selected_spoof is not None:
        paths["spoof_audio"] = str(selected_spoof)
    if selected_lid is not None:
        paths["lid_checkpoint"] = str(selected_lid)
    return task_config, out_dir, selected_spoof


def _copy_if_present(source: Path, target: Path) -> None:
    if source.exists() and source.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _organize_existing_artifacts(out_dir: Path) -> None:
    _copy_if_present(Path("outputs") / "anti_spoof_roc.png", out_dir / "anti_spoof_roc.png")
    _copy_if_present(Path("outputs") / "anti_spoof_confusion_matrix.png", out_dir / "anti_spoof_confusion_matrix.png")
    _copy_if_present(Path("outputs") / "anti_spoof_metrics.json", out_dir / "anti_spoof_metrics.json")
    _copy_if_present(Path("outputs") / "lid_fgsm_attack.wav", out_dir / "lid_fgsm_attack.wav")
    _copy_if_present(Path("models") / "anti_spoof_lfcc.pt", out_dir / "anti_spoof_lfcc.pt")


def _save_attack_curve_plot(attack_metrics: dict[str, object] | None, out_dir: Path) -> Path | None:
    if not attack_metrics or not attack_metrics.get("epsilon_trials"):
        return None

    import matplotlib.pyplot as plt

    trials = attack_metrics["epsilon_trials"]
    epsilons = [float(trial["epsilon"]) for trial in trials]
    snrs = [float(trial["snr_db"]) for trial in trials]
    target_ratios = [float(trial["target_frame_ratio"]) for trial in trials]
    successes = [bool(trial["success"]) for trial in trials]
    threshold = float(attack_metrics.get("snr_threshold_db", 40.0))

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(epsilons, target_ratios, marker="o", color="#4E79A7", label="Target frame ratio")
    ax1.set_xlabel("FGSM epsilon")
    ax1.set_ylabel("Target frame ratio", color="#4E79A7")
    ax1.tick_params(axis="y", labelcolor="#4E79A7")
    ax1.set_ylim(0.0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(epsilons, snrs, marker="s", color="#E15759", label="SNR (dB)")
    ax2.axhline(threshold, linestyle="--", linewidth=1.0, color="#59A14F", label=f"SNR threshold {threshold:.1f} dB")
    ax2.set_ylabel("SNR (dB)", color="#E15759")
    ax2.tick_params(axis="y", labelcolor="#E15759")

    for epsilon, ratio, success in zip(epsilons, target_ratios, successes):
        if success:
            ax1.scatter([epsilon], [ratio], color="#59A14F", s=70, zorder=5)

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="lower right")
    ax1.set_title("Task 4 FGSM Attack Sweep")
    fig.tight_layout()
    plot_path = out_dir / "fgsm_attack_curve.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _run_anti_spoof(config: dict, out_dir: Path, spoof_audio: Path | None, device: str) -> dict[str, object]:
    if spoof_audio is None or not spoof_audio.exists():
        return {
            "available": False,
            "reason": "No synthesized spoof audio was found. Run Task 3 first or pass --spoof-audio.",
        }

    manifest_summary = build_spoof_segment_manifest(
        bona_fide_audio=config["paths"]["student_voice_audio"],
        spoof_audio=spoof_audio,
        output_manifest=config["paths"]["spoof_eval_manifest"],
        sample_rate=config["audio"]["sample_rate"],
        chunk_seconds=float(config.get("reporting", {}).get("spoof_chunk_seconds", 4.0)),
        group_chunks=int(config.get("reporting", {}).get("spoof_group_chunks", 2)),
        max_chunks_per_label=int(config.get("reporting", {}).get("spoof_max_chunks_per_label", 60)),
    )
    spoof_metrics = evaluate_spoof_experiment(
        manifest_path=config["paths"]["spoof_eval_manifest"],
        checkpoint_path=config["paths"]["spoof_checkpoint"],
        sample_rate=config["audio"]["sample_rate"],
        n_lfcc=config["spoof"]["n_lfcc"],
        batch_size=config["spoof"]["batch_size"],
        epochs=config["spoof"]["epochs"],
        learning_rate=config["spoof"]["learning_rate"],
        device=device,
        roc_plot_path=out_dir / "anti_spoof_roc.png",
        confusion_plot_path=out_dir / "anti_spoof_confusion_matrix.png",
    )
    return {
        "available": True,
        **manifest_summary,
        **spoof_metrics,
    }


def _write_summary(
    config: dict,
    out_dir: Path,
    spoof_audio: Path | None,
    spoof_metrics: dict[str, object] | None,
    attack_metrics: dict[str, object] | None,
) -> Path:
    summary_path = out_dir / "task4_metrics.json"
    if spoof_metrics is None:
        copied_spoof_metrics = out_dir / "anti_spoof_metrics.json"
        if copied_spoof_metrics.exists():
            spoof_metrics = json.loads(copied_spoof_metrics.read_text(encoding="utf-8"))
    attack_curve_plot_path = _save_attack_curve_plot(attack_metrics, out_dir)

    summary = {
        "task": TASK_NAME,
        "entrypoint": "python tasks/task4_spoof_robustness/main.py",
        "results_dir": str(out_dir),
        "input_spoof_audio": str(spoof_audio) if spoof_audio else None,
        "core_code_files": [
            "src/assignment2/modules/spoof.py",
            "src/assignment2/modules/adversarial.py",
            "src/assignment2/modules/reporting.py",
        ],
        "outputs": {
            "spoof_manifest": config["paths"]["spoof_eval_manifest"],
            "anti_spoof_checkpoint": config["paths"]["spoof_checkpoint"],
            "roc_plot": str(out_dir / "anti_spoof_roc.png"),
            "confusion_matrix_plot": str(out_dir / "anti_spoof_confusion_matrix.png"),
            "fgsm_attack_curve_plot": str(attack_curve_plot_path) if attack_curve_plot_path else None,
            "fgsm_attack_audio": str(out_dir / "lid_fgsm_attack.wav"),
            "metrics": str(summary_path),
        },
        "metrics": {
            "spoof": spoof_metrics if spoof_metrics is not None else {"available": False, "reason": "Not run."},
            "adversarial": attack_metrics if attack_metrics is not None else {"available": False, "reason": "Not run."},
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=TASK_NAME)
    parser.add_argument("--config", default="configs/assignment2_config.json")
    parser.add_argument(
        "--stage",
        choices=["all", "anti-spoof", "attack", "evaluate", "summary"],
        default="evaluate",
        help="Default runs anti-spoof evaluation and FGSM evaluation when inputs are available.",
    )
    parser.add_argument("--spoof-audio", default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    os.chdir(ROOT)
    args = build_argparser().parse_args()
    config, out_dir, spoof_audio = _configure_task_paths(load_config(args.config), args.spoof_audio)
    device = _select_device(args.device)

    spoof_metrics = None
    attack_metrics = None
    if args.stage in {"all", "anti-spoof", "evaluate"}:
        spoof_metrics = _run_anti_spoof(config, out_dir, spoof_audio, device)
    if args.stage in {"all", "attack", "evaluate"}:
        command_attack(config, device)
        attack_metrics = evaluate_attack_metrics(config, device)

    _organize_existing_artifacts(out_dir)
    summary_path = _write_summary(config, out_dir, spoof_audio, spoof_metrics, attack_metrics)
    print(f"Wrote {TASK_NAME} summary to {summary_path}")


if __name__ == "__main__":
    main()
