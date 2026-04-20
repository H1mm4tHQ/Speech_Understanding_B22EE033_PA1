# Assignment 2 Report Outline

## 1. Introduction
- Problem statement
- Why code-switched Hinglish is hard
- Why Gujarati was selected as the target LRL

## 2. Part I: Robust Code-Switched Transcription
- Main program: `tasks/task1_robust_stt/main.py`
- Results folder: `outputs/task1/`
- Denoising and normalization
- Multi-head frame-level LID architecture
- Training data and supervision format
- Constrained decoding with N-gram LM
- Evaluation: WER, LID F1, switch-boundary confusion matrix

## 3. Part II: Phonetic Mapping and Translation
- Main program: `tasks/task2_ipa_translation/main.py`
- Results folder: `outputs/task2/`
- Unified IPA design for Hinglish phonology
- Manual mapping strategy for Devanagari and Roman tokens
- Gujarati translation corpus construction
- Parallel dictionary evidence: `outputs/task2/technical_parallel_dictionary.tsv`
- Corpus stats: `outputs/task2/parallel_corpus_stats.json`
- Examples from the lecture excerpt

## 4. Part III: Cross-Lingual Voice Cloning
- Main program: `tasks/task3_voice_cloning/main.py`
- Results folder: `outputs/task3/`
- Student reference recording and x-vector extraction
- Prosody extraction and DTW warping
- Gujarati TTS backend and synthesis flow
- Cleaned audio output: `outputs/task3/output_LRL_cloned.wav`
- Raw audio retained for comparison: `outputs/task3/output_LRL_cloned_raw.wav`
- Ablation: flat synthesis vs DTW-warped synthesis

## 5. Part IV: Robustness and Spoof Detection
- Main program: `tasks/task4_spoof_robustness/main.py`
- Results folder: `outputs/task4/`
- LFCC anti-spoofing model
- EER computation
- FGSM attack setup
- Minimum epsilon under SNR > 40 dB

## 6. Results
- Tables for WER, F1, switch accuracy, MCD, EER, epsilon
- Audio examples and qualitative observations

## 7. References
- Cite PyTorch, torchaudio, transformers, and any pretrained checkpoints used
