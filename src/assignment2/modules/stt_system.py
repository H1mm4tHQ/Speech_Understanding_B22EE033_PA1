from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.generation import LogitsProcessor, LogitsProcessorList

from assignment2.modules.ctc_decode import ConstrainedCTCBeamSearch
from assignment2.modules.ngram_lm import NGramLanguageModel, TERM_PATTERN
from assignment2.utils.audio import chunk_audio, load_audio


def _resolve_model_source(model_name: str) -> tuple[str | Path, bool]:
    if Path(model_name).exists():
        return Path(model_name), True
    repo_dir_name = f"models--{model_name.replace('/', '--')}"
    cache_roots = [
        Path.cwd() / ".hf_cache" / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for cache_root in cache_roots:
        snapshots_dir = cache_root / repo_dir_name / "snapshots"
        if not snapshots_dir.exists():
            continue
        for snapshot_dir in sorted(snapshots_dir.iterdir(), reverse=True):
            has_model = any((snapshot_dir / name).exists() for name in ["model.safetensors", "pytorch_model.bin"])
            has_processor = any(
                (snapshot_dir / name).exists()
                for name in ["preprocessor_config.json", "processor_config.json", "tokenizer_config.json"]
            )
            if has_model and has_processor:
                return snapshot_dir, True
    return model_name, False


def _normalize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.replace("  ", " ").strip().split())


def _comparison_token(token: str) -> str:
    return re.sub(r"[^\w\u0900-\u097F]+", "", token.casefold())


def _collapse_repeated_phrases(words: list[str], max_ngram: int = 18) -> list[str]:
    current = words
    changed = True
    while changed:
        changed = False
        output: list[str] = []
        index = 0
        while index < len(current):
            collapsed = False
            max_size = min(max_ngram, (len(current) - index) // 2)
            for size in range(max_size, 0, -1):
                left = current[index : index + size]
                right = current[index + size : index + (2 * size)]
                if not left or not right:
                    continue
                left_cmp = [_comparison_token(token) for token in left]
                right_cmp = [_comparison_token(token) for token in right]
                if any(left_cmp) and left_cmp == right_cmp:
                    output.extend(left)
                    index += 2 * size
                    changed = True
                    collapsed = True
                    break
            if collapsed:
                continue
            output.append(current[index])
            index += 1
        current = output
    return current


def _finalize_merged_text(text: str) -> str:
    words = [part for part in _normalize_text(text).split() if part]
    words = _collapse_repeated_phrases(words)
    return " ".join(words).strip()


def _extract_technical_terms(syllabus_text: str) -> list[str]:
    normalized_lines = [_normalize_text(line).casefold() for line in syllabus_text.splitlines() if line.strip()]
    extracted_terms: set[str] = set(normalized_lines)
    extracted_terms.update(match.group(0).casefold() for match in TERM_PATTERN.finditer(syllabus_text))
    return sorted((term for term in extracted_terms if term), key=len, reverse=True)


class SyllabusBiasLogitsProcessor(LogitsProcessor):
    """Adds a small bonus to technical-term tokens and a stronger bonus to matching prefixes."""

    def __init__(
        self,
        tokenizer,
        technical_terms: list[str],
        initial_token_bias: float = 0.25,
        prefix_token_bias: float = 1.5,
    ) -> None:
        self.initial_token_bias = float(initial_token_bias)
        self.prefix_token_bias = float(prefix_token_bias)
        self.initial_token_scores: dict[int, float] = defaultdict(float)
        self.prefix_to_next_token: dict[tuple[int, ...], dict[int, float]] = defaultdict(dict)
        self.max_prefix_length = 0

        seen_sequences: set[tuple[int, ...]] = set()
        for term in technical_terms:
            candidate_forms = {term, f" {term}"}
            for form in candidate_forms:
                token_ids = tuple(tokenizer.encode(form, add_special_tokens=False))
                if not token_ids or token_ids in seen_sequences:
                    continue
                seen_sequences.add(token_ids)
                self.initial_token_scores[token_ids[0]] = max(
                    self.initial_token_scores.get(token_ids[0], 0.0),
                    self.initial_token_bias,
                )
                for prefix_len in range(1, len(token_ids)):
                    prefix = token_ids[:prefix_len]
                    next_token = token_ids[prefix_len]
                    self.max_prefix_length = max(self.max_prefix_length, prefix_len)
                    existing = self.prefix_to_next_token[prefix].get(next_token, 0.0)
                    self.prefix_to_next_token[prefix][next_token] = max(existing, self.prefix_token_bias)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.initial_token_scores:
            for token_id, bonus in self.initial_token_scores.items():
                scores[:, token_id] += bonus

        if self.max_prefix_length <= 0:
            return scores

        for row_index, sequence in enumerate(input_ids.tolist()):
            suffix = sequence[-self.max_prefix_length :]
            for prefix_len in range(1, min(len(suffix), self.max_prefix_length) + 1):
                prefix = tuple(suffix[-prefix_len:])
                for token_id, bonus in self.prefix_to_next_token.get(prefix, {}).items():
                    scores[row_index, token_id] += bonus
        return scores


class ConstrainedCTCTranscriber:
    def __init__(
        self,
        model_name: str,
        syllabus_text: str,
        ngram_order: int,
        beam_size: int,
        lm_weight: float,
        length_penalty: float,
        term_bonus: float,
        device: str,
    ) -> None:
        model_source, local_only = _resolve_model_source(model_name)
        self.processor = AutoProcessor.from_pretrained(model_source, local_files_only=local_only)
        self.model = AutoModelForCTC.from_pretrained(model_source, local_files_only=local_only).to(device)
        self.model.eval()
        self.device = device

        tokenizer = self.processor.tokenizer
        vocab = tokenizer.get_vocab()
        self.id_to_token = {idx: token for token, idx in vocab.items()}
        blank_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        self.language_model = NGramLanguageModel(order=ngram_order)
        self.language_model.fit_from_text(syllabus_text, tokenizer=tokenizer)
        self.decoder = ConstrainedCTCBeamSearch(
            beam_size=beam_size,
            lm_weight=lm_weight,
            length_penalty=length_penalty,
            term_bonus=term_bonus,
            blank_id=blank_id,
            ngram_lm=self.language_model,
            id_to_token=self.id_to_token,
        )

    @property
    def sample_rate(self) -> int:
        return self.processor.feature_extractor.sampling_rate

    def transcribe_chunk(self, waveform: torch.Tensor, sample_rate: int) -> str:
        inputs = self.processor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0].cpu()
        text, _ = self.decoder.decode(logits)
        return _normalize_text(text)

    def _merge_chunk_texts(self, texts: list[str], max_overlap_words: int = 12) -> str:
        merged_words: list[str] = []
        for index, text in enumerate(texts):
            words = [part for part in _normalize_text(text).split() if part]
            if not words:
                continue
            if index == 0 or not merged_words:
                merged_words.extend(words)
                continue
            max_overlap = min(max_overlap_words, len(merged_words), len(words))
            overlap = 0
            for size in range(max_overlap, 0, -1):
                if merged_words[-size:] == words[:size]:
                    overlap = size
                    break
            merged_words.extend(words[overlap:])
        return _finalize_merged_text(" ".join(merged_words))

    def transcribe_file(self, audio_path: str | Path, chunk_seconds: float, overlap_seconds: float) -> str:
        waveform, sample_rate = load_audio(audio_path, target_sr=self.sample_rate, mono=True)
        chunks = chunk_audio(
            waveform,
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )
        texts = [self.transcribe_chunk(chunk, sample_rate) for _, _, chunk in chunks]
        return self._merge_chunk_texts(texts)


class ConstrainedWhisperTranscriber:
    def __init__(
        self,
        model_name: str,
        syllabus_text: str,
        beam_size: int,
        length_penalty: float,
        term_bonus: float,
        initial_token_bias: float,
        max_new_tokens: int,
        language: str | None,
        task: str,
        device: str,
    ) -> None:
        model_source, local_only = _resolve_model_source(model_name)
        self.processor = AutoProcessor.from_pretrained(model_source, local_files_only=local_only)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_source, local_files_only=local_only).to(device)
        self.model.eval()
        self.device = device
        self.language = language
        self.task = task
        self.beam_size = beam_size
        self.length_penalty = max(float(length_penalty), 0.0)
        self.max_new_tokens = int(max_new_tokens)

        technical_terms = _extract_technical_terms(syllabus_text)
        self.logits_processor = LogitsProcessorList()
        if technical_terms:
            self.logits_processor.append(
                SyllabusBiasLogitsProcessor(
                    tokenizer=self.processor.tokenizer,
                    technical_terms=technical_terms,
                    initial_token_bias=initial_token_bias,
                    prefix_token_bias=term_bonus,
                )
            )

    @property
    def sample_rate(self) -> int:
        return self.processor.feature_extractor.sampling_rate

    def transcribe_chunk(self, waveform: torch.Tensor, sample_rate: int) -> str:
        inputs = self.processor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                language=self.language,
                task=self.task,
                num_beams=self.beam_size,
                length_penalty=self.length_penalty,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
                logits_processor=self.logits_processor if len(self.logits_processor) > 0 else None,
                return_timestamps=False,
            )
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _normalize_text(text)

    def _merge_chunk_texts(self, texts: list[str], max_overlap_words: int = 18) -> str:
        merged_words: list[str] = []
        for index, text in enumerate(texts):
            words = [part for part in _normalize_text(text).split() if part]
            if not words:
                continue
            if index == 0 or not merged_words:
                merged_words.extend(words)
                continue
            max_overlap = min(max_overlap_words, len(merged_words), len(words))
            overlap = 0
            for size in range(max_overlap, 0, -1):
                if merged_words[-size:] == words[:size]:
                    overlap = size
                    break
            merged_words.extend(words[overlap:])
        return _finalize_merged_text(" ".join(merged_words))

    def transcribe_file(self, audio_path: str | Path, chunk_seconds: float, overlap_seconds: float) -> str:
        waveform, sample_rate = load_audio(audio_path, target_sr=self.sample_rate, mono=True)
        chunks = chunk_audio(
            waveform,
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )
        texts = [self.transcribe_chunk(chunk, sample_rate) for _, _, chunk in chunks]
        return self._merge_chunk_texts(texts)


def build_transcriber(
    model_name: str,
    syllabus_text: str,
    ngram_order: int,
    beam_size: int,
    lm_weight: float,
    length_penalty: float,
    term_bonus: float,
    device: str,
    backend: str = "auto",
    language: str | None = None,
    task: str = "transcribe",
    initial_token_bias: float = 0.25,
    max_new_tokens: int = 256,
):
    model_source, local_only = _resolve_model_source(model_name)
    use_whisper = backend == "whisper"
    if backend == "auto":
        config = AutoConfig.from_pretrained(model_source, local_files_only=local_only)
        use_whisper = getattr(config, "model_type", "") == "whisper"

    if use_whisper:
        return ConstrainedWhisperTranscriber(
            model_name=model_name,
            syllabus_text=syllabus_text,
            beam_size=beam_size,
            length_penalty=length_penalty,
            term_bonus=term_bonus,
            initial_token_bias=initial_token_bias,
            max_new_tokens=max_new_tokens,
            language=language,
            task=task,
            device=device,
        )

    return ConstrainedCTCTranscriber(
        model_name=model_name,
        syllabus_text=syllabus_text,
        ngram_order=ngram_order,
        beam_size=beam_size,
        lm_weight=lm_weight,
        length_penalty=length_penalty,
        term_bonus=term_bonus,
        device=device,
    )
