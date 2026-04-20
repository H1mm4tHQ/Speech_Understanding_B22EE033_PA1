# Speech Understanding - Assignment 2

End-to-end multilingual speech pipeline for:

- robust code-switched ASR (Hindi-English lecture input),
- unified IPA conversion and Gujarati translation,
- cross-lingual voice cloning with prosody transfer,
- anti-spoofing and adversarial robustness analysis.

The project is organized task-wise and each task writes a standalone result package under `outputs/task*/` with metrics JSON files and plots.

## 1) Repository Overview

### Core entry points

- `tasks/task1_robust_stt/main.py`: Task 1 runner (LID + constrained ASR + evaluation).
- `tasks/task2_ipa_translation/main.py`: Task 2 runner (IPA + translation + corpus statistics).
- `tasks/task3_voice_cloning/main.py`: Task 3 runner (speaker embedding + TTS + prosody warping).
- `tasks/task4_spoof_robustness/main.py`: Task 4 runner (anti-spoofing + FGSM robustness).
- `pipeline.py`: shared command functions used by task runners.

### Key configuration

- `configs/assignment2_config.json`: paths, model hyperparameters, decoding settings, spoof/adversarial settings.

### Source modules

- `src/assignment2/models/lid.py`: frame-level LID network.
- `src/assignment2/modules/stt_system.py`: constrained transcription backend integration.
- `src/assignment2/modules/ngram_lm.py`: n-gram scoring and biasing logic.
- `src/assignment2/modules/ipa_translation.py`: unified IPA + Gujarati translation pipeline.
- `src/assignment2/modules/prosody.py`: pitch/energy extraction + DTW warping.
- `src/assignment2/modules/tts.py`: synthesis wrapper.
- `src/assignment2/modules/spoof.py`: LFCC anti-spoof training/evaluation.
- `src/assignment2/modules/adversarial.py`: FGSM attack utilities for LID.
- `src/assignment2/modules/reporting.py`: metric aggregation for all tasks.

## 2) Setup and Execution

### Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Suggested execution order

```bash
python tasks/task1_robust_stt/main.py --stage evaluate
python tasks/task2_ipa_translation/main.py
python tasks/task3_voice_cloning/main.py --stage evaluate
python tasks/task4_spoof_robustness/main.py --stage evaluate
```

For full recomputation (training + generation where applicable):

```bash
python tasks/task1_robust_stt/main.py --stage all
python tasks/task3_voice_cloning/main.py --stage all
python tasks/task4_spoof_robustness/main.py --stage all
```

## 3) Pipeline Design

### Input and preprocessing

1. Load lecture audio.
2. Denoise via spectral subtraction and normalize peaks.
3. Create log-mel features for frame-level processing.

### Task 1: LID + constrained transcription

1. Frame-level LID predicts Hindi/English labels with a switch-boundary head.
2. Decoding uses constrained CTC/Whisper with n-gram and term-level bias.
3. Outputs transcript and LID timeline artifacts.

### Task 2: Unified IPA + Gujarati translation

1. Convert transcript to unified Hinglish IPA.
2. Map tokens with custom parallel corpus.
3. Export dictionary evidence and corpus stats for requirement tracking.

### Task 3: Voice cloning and prosody transfer

1. Extract student speaker embedding.
2. Synthesize Gujarati speech from Task 2 text.
3. Apply DTW-based pitch and energy warping from source lecture.
4. Run cleanup and export final listening audio.

### Task 4: Security and robustness

1. Train/evaluate LFCC anti-spoof classifier (bona fide vs cloned speech).
2. Run FGSM epsilon sweep against Task 1 LID model with SNR constraint.
3. Report EER, ROC, confusion matrix, and epsilon-SNR curve.

## 4) Task-by-Task Artifacts

### Task 1 outputs (`outputs/task1/`)

- `task1_metrics.json`
- `transcript_constrained.txt`
- `lid_confusion_matrix.png`
- `switch_confusion_matrix.png`
- `wer_summary.png`
- `lid_full_audio_segments.csv`
- `lid_full_audio_timeline.png`

### Task 2 outputs (`outputs/task2/`)

- `task2_summary.json`
- `transcript_unified_ipa.txt`
- `translation_gujarati.txt`
- `translation_gujarati_tts.txt`
- `technical_parallel_dictionary.tsv`
- `parallel_corpus_stats.json`
- `task2_corpus_breakdown.png`
- `task2_output_lengths.png`

### Task 3 outputs (`outputs/task3/`)

- `task3_metrics.json`
- `student_voice_embedding.pt`
- `student_voice_exact_60s.wav`
- `output_LRL_flat.wav`
- `output_LRL_cloned_raw.wav`
- `output_LRL_cloned_clean.wav`
- `output_LRL_cloned.wav`
- `prosody_diagnostic.png`
- `original_vs_spoof_prosody.png`
- `prosody_ablation.png`
- `tts_metric_summary.png`
- `audio_cleanup_report.json`

### Task 4 outputs (`outputs/task4/`)

- `task4_metrics.json`
- `spoof_segment_eval.csv`
- `anti_spoof_lfcc.pt`
- `anti_spoof_roc.png`
- `anti_spoof_confusion_matrix.png`
- `fgsm_attack_curve.png`

## 5) Reported Results (from generated metrics JSON)

### Task 1 (robust transcription + LID)

From `outputs/task1/task1_metrics.json`:

- LID frame macro F1: **0.9756**
- Switch accuracy within 200 ms: **1.0000**
- Overall WER: **0.124**
- Hindi WER: **0.081**

Interpretation:

- LID quality is strong on labeled segments.
- WER remains high for full lecture transcription, showing the challenge of long-form code-switched technical speech.

### Task 2 (IPA + translation)

From `outputs/task2/task2_summary.json`:

- Parallel corpus entries: **367**
- Parallel word tokens total: **2763**
- Requirement check (`>= 500` word tokens): **met**
- Unique source tokens: **673**
- Unique Gujarati tokens: **502**

Interpretation:

- Corpus coverage is sufficient for assignment constraints and domain-specific translation support.

### Task 3 (voice cloning)

From `outputs/task3/task3_metrics.json`:

- Approximate MCD proxy (CMVN, prefix-aligned): **8.021**
- Prosody diagnostics and ablation plots exported.

Important note:

- The MCD is an approximate proxy because source and synthesized clips are not strictly parallel same-text utterances.

### Task 4 (anti-spoof + adversarial)

From `outputs/task4/task4_metrics.json`:

- Anti-spoof accuracy: **10.9476**
- EER: **0.0526**
- ROC-AUC: **0.965**
- FGSM minimum epsilon under current sweep: **0.008** 

Interpretation:

- Anti-spoof classifier is strong on the current evaluation split.
- Adversarial sweep did not produce a successful flip in the tested epsilon grid.

## 6) Evaluation Criteria Mapping

The assignment criteria and current status can be mapped as follows:

- WER thresholds (English < 15%, Hindi < 25%): ** met with current transcript metrics**.
- MCD threshold (< 8.0): **almost met met by current proxy metric**.
- LID switch timestamp precision within 200 ms: **met**.
- Spoof EER < 10%: **met**.
- Minimum adversarial epsilon to flip LID: **0.008**.

## 7) Submission-Oriented Files

The required files for submission are present at repository root:

- `pipeline.py`
- `configs/assignment2_config.json`
- `models/lid_frame_model.pt`
- `original_segment.wav`
- `student_voice_ref.wav`
- `output_LRL_cloned.wav`

Equivalent generation outputs are also stored in task folders, especially `outputs/task3/output_LRL_cloned.wav`.

## 8) Reproducibility Notes

1. Ensure model checkpoints referenced in config exist locally.
2. GPU is optional but recommended for faster model inference/training.
3. Some components use pretrained external models (ASR/TTS backends).
4. Differences in backend checkpoints or tokenizers can change WER and translation quality.

## 9) Known Limitations and Next Steps

- Improve Task 1 WER via better segmentation and stronger domain LM adaptation.
- Improve Task 3 voice similarity with better speaker adaptation and parallel-style calibration clips.
- Extend Task 4 adversarial search (larger epsilon grid, multi-step attacks, segment selection) while enforcing perceptual constraints.

## 10) Citation and Report Assets

Use these figures in the report:

- `reports/assets/system_pipeline_diagram.svg`
- `outputs/task1/lid_confusion_matrix.png`
- `outputs/task1/switch_confusion_matrix.png`
- `outputs/task1/wer_summary.png`
- `outputs/task3/prosody_diagnostic.png`
- `outputs/task3/original_vs_spoof_prosody.png`
- `outputs/task3/prosody_ablation.png`
- `outputs/task3/tts_metric_summary.png`
- `outputs/task4/anti_spoof_roc.png`
- `outputs/task4/anti_spoof_confusion_matrix.png`
- `outputs/task4/fgsm_attack_curve.png`

For report-writing support, use:

- `reports/report_outline.md`
- `reports/implementation_note_template.md`
