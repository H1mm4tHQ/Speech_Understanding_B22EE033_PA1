from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assignment2.modules.ipa_translation import CorpusTranslator, UnifiedIPAMapper, normalize_text
from pipeline import load_config


TASK_NAME = "Task 2 - IPA Mapping and Gujarati Translation"
TASK_DIR = Path("outputs") / "task2"
WORD_PATTERN = re.compile(r"[A-Za-z0-9_']+|[\u0900-\u097F]+", re.UNICODE)
CORPUS_WORD_PATTERN = re.compile(r"[A-Za-z0-9_']+|[\u0900-\u097F]+|[\u0A80-\u0AFF]+", re.UNICODE)
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n", re.UNICODE)


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("No transcript input was found for Task 2.")


def _configure_task_paths(config: dict) -> tuple[dict, Path]:
    task_config = copy.deepcopy(config)
    out_dir = TASK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    task_config["paths"]["outputs_dir"] = str(out_dir)
    task_config["paths"]["translation_output"] = str(out_dir / "translation_gujarati.txt")
    return task_config, out_dir


def _default_input() -> Path:
    return _first_existing(
        [
            Path("outputs") / "task1" / "transcript_constrained.txt",
            Path("outputs") / "task1" / "manual_transcript_excerpt.txt",
            Path("outputs") / "transcript_constrained.txt",
            Path("outputs") / "manual_transcript_excerpt.txt",
        ]
    )


def _load_segment_overrides(path: str | Path) -> dict[str, str]:
    override_path = Path(path)
    if not override_path.exists():
        return {}
    entries = json.loads(override_path.read_text(encoding="utf-8"))
    return {
        normalize_text(entry["source_text"]): entry["gujarati_translation"].strip()
        for entry in entries
        if entry.get("source_text", "").strip() and entry.get("gujarati_translation", "").strip()
    }


def _translate_with_overrides(text: str, translator: CorpusTranslator, overrides: dict[str, str]) -> str:
    paragraphs = [part.strip() for part in PARAGRAPH_SPLIT_PATTERN.split(text) if part.strip()]
    translated_paragraphs: list[str] = []
    for paragraph in paragraphs:
        translated_paragraphs.append(overrides.get(normalize_text(paragraph), translator.translate(paragraph)))
    return "\n\n".join(part for part in translated_paragraphs if part.strip())


def _prepare_tts_text(text: str) -> str:
    lines: list[str] = []
    for paragraph in [part.strip() for part in PARAGRAPH_SPLIT_PATTERN.split(text) if part.strip()]:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        sentence_candidates = [part.strip() for part in re.split(r"(?<=[.!?।])\s+", paragraph) if part.strip()]
        normalized_sentences: list[str] = []
        for sentence in sentence_candidates:
            words = sentence.split()
            if len(words) <= 22:
                normalized_sentences.append(sentence)
                continue
            chunk: list[str] = []
            for word in words:
                chunk.append(word)
                if len(chunk) >= 18 and word.endswith((",", ";", ":")):
                    normalized_sentences.append(" ".join(chunk).strip(" ,;:"))
                    chunk = []
            if chunk:
                while len(chunk) > 22:
                    normalized_sentences.append(" ".join(chunk[:22]))
                    chunk = chunk[22:]
                if chunk:
                    normalized_sentences.append(" ".join(chunk))
        lines.extend(sentence if sentence.endswith((".", "!", "?", "।")) else f"{sentence}." for sentence in normalized_sentences)
        lines.append("")
    return "\n".join(lines).strip()


def _write_parallel_corpus_artifacts(config: dict, out_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    corpus_path = Path(config["paths"]["task2_corpus"])
    dictionary_path = out_dir / "technical_parallel_dictionary.tsv"
    stats_path = out_dir / "parallel_corpus_stats.json"

    rows: list[dict[str, str]] = []
    with corpus_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    fieldnames = [
        "entry_id",
        "source_text",
        "alt_source_text",
        "source_lang",
        "unified_ipa",
        "gujarati_translation",
        "category",
        "notes",
    ]
    with dictionary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    source_tokens: list[str] = []
    gujarati_tokens: list[str] = []
    ipa_entries = 0
    category_counts: dict[str, int] = {}
    for row in rows:
        source_tokens.extend(CORPUS_WORD_PATTERN.findall(row.get("source_text", "")))
        source_tokens.extend(CORPUS_WORD_PATTERN.findall(row.get("alt_source_text", "")))
        gujarati_tokens.extend(CORPUS_WORD_PATTERN.findall(row.get("gujarati_translation", "")))
        if row.get("unified_ipa", "").strip():
            ipa_entries += 1
        category = row.get("category", "").strip() or "uncategorized"
        category_counts[category] = category_counts.get(category, 0) + 1

    stats = {
        "source_corpus": str(corpus_path),
        "exported_dictionary": str(dictionary_path),
        "entry_count": len(rows),
        "manual_ipa_mapping_entries": ipa_entries,
        "source_word_tokens": len(source_tokens),
        "unique_source_word_tokens": len({token.casefold() for token in source_tokens}),
        "gujarati_word_tokens": len(gujarati_tokens),
        "unique_gujarati_word_tokens": len(set(gujarati_tokens)),
        "parallel_word_tokens_total": len(source_tokens) + len(gujarati_tokens),
        "meets_500_word_parallel_corpus_requirement": (len(source_tokens) + len(gujarati_tokens)) >= 500,
        "category_counts": category_counts,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    shutil.copy2(corpus_path, out_dir / corpus_path.name)
    return dictionary_path, stats_path, stats


def _save_task2_plots(
    summary_counts: dict[str, int],
    corpus_stats: dict[str, object],
    out_dir: Path,
) -> tuple[Path | None, Path | None]:
    import matplotlib.pyplot as plt

    corpus_plot_path: Path | None = None
    counts_plot_path: Path | None = None

    source_tokens = int(corpus_stats.get("source_word_tokens", 0))
    gujarati_tokens = int(corpus_stats.get("gujarati_word_tokens", 0))
    total_tokens = int(corpus_stats.get("parallel_word_tokens_total", 0))
    if total_tokens > 0:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        labels = ["Source", "Gujarati", "Total", "Target"]
        values = [source_tokens, gujarati_tokens, total_tokens, 500]
        colors = ["#4E79A7", "#59A14F", "#F28E2B", "#BAB0AC"]
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel("Token count")
        ax.set_title("Task 2 Parallel Corpus Coverage")
        ax.axhline(500, linestyle="--", linewidth=1.0, color="#E15759", label="500-word requirement")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(total_tokens * 0.01, 4), str(value), ha="center")
        ax.legend(loc="upper left")
        fig.tight_layout()
        corpus_plot_path = out_dir / "task2_corpus_breakdown.png"
        fig.savefig(corpus_plot_path, dpi=180)
        plt.close(fig)

    source_count = int(summary_counts.get("source_tokens", 0))
    ipa_count = int(summary_counts.get("ipa_characters", 0))
    gujarati_count = int(summary_counts.get("gujarati_characters", 0))
    if max(source_count, ipa_count, gujarati_count) > 0:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        labels = ["Transcript tokens", "IPA chars", "Gujarati chars"]
        values = [source_count, ipa_count, gujarati_count]
        bars = ax.bar(labels, values, color=["#4E79A7", "#9C755F", "#59A14F"])
        ax.set_ylabel("Count")
        ax.set_title("Task 2 Output Size Summary")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(max(values) * 0.01, 2), str(value), ha="center")
        fig.tight_layout()
        counts_plot_path = out_dir / "task2_output_lengths.png"
        fig.savefig(counts_plot_path, dpi=180)
        plt.close(fig)

    return corpus_plot_path, counts_plot_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=TASK_NAME)
    parser.add_argument("--config", default="configs/assignment2_config.json")
    parser.add_argument("--text-file", default=None, help="Transcript to map and translate.")
    return parser


def main() -> None:
    os.chdir(ROOT)
    args = build_argparser().parse_args()
    config, out_dir = _configure_task_paths(load_config(args.config))
    input_path = Path(args.text_file) if args.text_file else _default_input()
    text = input_path.read_text(encoding="utf-8")

    mapper = UnifiedIPAMapper(config["paths"]["task2_corpus"])
    translator = CorpusTranslator(config["paths"]["task2_corpus"])
    overrides = _load_segment_overrides("data/task2_segment_overrides.json")
    ipa = mapper.text_to_ipa(text)
    gujarati = _translate_with_overrides(text, translator, overrides)
    gujarati_tts = _prepare_tts_text(gujarati)

    ipa_path = out_dir / "transcript_unified_ipa.txt"
    translation_path = out_dir / "translation_gujarati.txt"
    tts_translation_path = out_dir / "translation_gujarati_tts.txt"
    summary_path = out_dir / "task2_summary.json"
    dictionary_path, corpus_stats_path, corpus_stats = _write_parallel_corpus_artifacts(config, out_dir)
    ipa_path.write_text(ipa, encoding="utf-8")
    translation_path.write_text(gujarati, encoding="utf-8")
    tts_translation_path.write_text(gujarati_tts, encoding="utf-8")

    counts = {
        "source_tokens": len(WORD_PATTERN.findall(text)),
        "ipa_characters": len(ipa),
        "gujarati_characters": len(gujarati),
    }
    corpus_plot_path, counts_plot_path = _save_task2_plots(counts, corpus_stats, out_dir)

    summary = {
        "task": TASK_NAME,
        "entrypoint": "python tasks/task2_ipa_translation/main.py",
        "results_dir": str(out_dir),
        "input_transcript": str(input_path),
        "core_code_files": [
            "src/assignment2/modules/ipa_translation.py",
            "mapping/task2_unified_hinglish_ipa_gujarati_corpus.tsv",
        ],
        "outputs": {
            "ipa_transcript": str(ipa_path),
            "gujarati_translation": str(translation_path),
            "gujarati_translation_tts_ready": str(tts_translation_path),
            "parallel_dictionary": str(dictionary_path),
            "parallel_corpus_stats": str(corpus_stats_path),
            "parallel_corpus_plot": str(corpus_plot_path) if corpus_plot_path else None,
            "output_lengths_plot": str(counts_plot_path) if counts_plot_path else None,
            "summary": str(summary_path),
        },
        "counts": counts,
        "parallel_corpus": corpus_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {TASK_NAME} outputs to {out_dir}")


if __name__ == "__main__":
    main()
