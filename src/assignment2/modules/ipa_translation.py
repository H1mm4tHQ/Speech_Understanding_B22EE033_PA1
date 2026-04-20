from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_']+|[\u0900-\u097F]+|[\u0A80-\u0AFF]+|[^\w\s]", re.UNICODE)
WORD_PATTERN = re.compile(r"[A-Za-z0-9_']+|[\u0900-\u097F]+|[\u0A80-\u0AFF]+", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?।])\s+|\n+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[“”\"'`]+", "", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def word_tokens(text: str) -> list[str]:
    return WORD_PATTERN.findall(normalize_text(text))


DEVANAGARI_VOWELS = {
    "अ": "ə",
    "आ": "aː",
    "इ": "ɪ",
    "ई": "iː",
    "उ": "ʊ",
    "ऊ": "uː",
    "ए": "eː",
    "ऐ": "ɛː",
    "ओ": "oː",
    "औ": "ɔː",
}

DEVANAGARI_MATRAS = {
    "ा": "aː",
    "ि": "ɪ",
    "ी": "iː",
    "ु": "ʊ",
    "ू": "uː",
    "े": "eː",
    "ै": "ɛː",
    "ो": "oː",
    "ौ": "ɔː",
    "ृ": "ɾɪ",
}

DEVANAGARI_CONSONANTS = {
    "क": "k",
    "ख": "kʰ",
    "ग": "ɡ",
    "घ": "ɡʱ",
    "च": "tʃ",
    "छ": "tʃʰ",
    "ज": "dʒ",
    "झ": "dʒʱ",
    "ट": "ʈ",
    "ठ": "ʈʰ",
    "ड": "ɖ",
    "ढ": "ɖʱ",
    "त": "t̪",
    "थ": "t̪ʰ",
    "द": "d̪",
    "ध": "d̪ʱ",
    "न": "n",
    "प": "p",
    "फ": "pʰ",
    "ब": "b",
    "भ": "bʱ",
    "म": "m",
    "य": "j",
    "र": "ɾ",
    "ल": "l",
    "व": "ʋ",
    "स": "s",
    "श": "ʃ",
    "ष": "ʂ",
    "ह": "ɦ",
    "ञ": "ɲ",
    "ण": "ɳ",
}

ROMAN_RULES = [
    ("tion", "ʃən"),
    ("ture", "tʃəɾ"),
    ("ph", "f"),
    ("gh", "ɡ"),
    ("ch", "tʃ"),
    ("sh", "ʃ"),
    ("th", "θ"),
    ("oo", "uː"),
    ("ee", "iː"),
    ("ai", "eɪ"),
    ("au", "ɔː"),
]

ROMAN_CHARS = {
    "a": "ə",
    "b": "b",
    "c": "k",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "ɡ",
    "h": "h",
    "i": "ɪ",
    "j": "dʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "p": "p",
    "q": "k",
    "r": "ɾ",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "x": "ks",
    "y": "j",
    "z": "z",
}

DEVANAGARI_TO_GUJARATI = {
    "अ": "અ",
    "आ": "આ",
    "इ": "ઇ",
    "ई": "ઈ",
    "उ": "ઉ",
    "ऊ": "ઊ",
    "ए": "એ",
    "ऐ": "ઐ",
    "ओ": "ઓ",
    "औ": "ઔ",
    "ऑ": "ઓ",
    "ऋ": "ઋ",
    "क": "ક",
    "ख": "ખ",
    "ग": "ગ",
    "घ": "ઘ",
    "ङ": "ઙ",
    "च": "ચ",
    "छ": "છ",
    "ज": "જ",
    "झ": "ઝ",
    "ञ": "ઞ",
    "ट": "ટ",
    "ठ": "ઠ",
    "ड": "ડ",
    "ढ": "ઢ",
    "ण": "ણ",
    "त": "ત",
    "थ": "થ",
    "द": "દ",
    "ध": "ધ",
    "न": "ન",
    "प": "પ",
    "फ": "ફ",
    "ब": "બ",
    "भ": "ભ",
    "म": "મ",
    "य": "ય",
    "र": "ર",
    "ल": "લ",
    "व": "વ",
    "श": "શ",
    "ष": "ષ",
    "स": "સ",
    "ह": "હ",
    "ा": "ા",
    "ि": "િ",
    "ी": "ી",
    "ु": "ુ",
    "ू": "ૂ",
    "े": "ે",
    "ै": "ૈ",
    "ो": "ો",
    "ौ": "ૌ",
    "ृ": "ૃ",
    "ॅ": "ે",
    "ॉ": "ો",
    "़": "",
    "ं": "ં",
    "ँ": "ં",
    "ः": "ઃ",
    "्": "્",
    "।": ".",
    "०": "૦",
    "१": "૧",
    "२": "૨",
    "३": "૩",
    "४": "૪",
    "५": "૫",
    "६": "૬",
    "७": "૭",
    "८": "૮",
    "९": "૯",
}

ROMAN_TO_GUJARATI_RULES = [
    ("tion", "શન"),
    ("sion", "ઝન"),
    ("sh", "શ"),
    ("ch", "ચ"),
    ("th", "થ"),
    ("dh", "ધ"),
    ("ph", "ફ"),
    ("bh", "ભ"),
    ("gh", "ઘ"),
    ("kh", "ખ"),
    ("aa", "આ"),
    ("ee", "ઈ"),
    ("ii", "ઈ"),
    ("oo", "ઊ"),
    ("uu", "ઊ"),
    ("ai", "ઐ"),
    ("au", "ઔ"),
    ("ng", "ંગ"),
]

ROMAN_TO_GUJARATI_CHARS = {
    "a": "અ",
    "b": "બ",
    "c": "ક",
    "d": "દ",
    "e": "એ",
    "f": "ફ",
    "g": "ગ",
    "h": "હ",
    "i": "ઇ",
    "j": "જ",
    "k": "ક",
    "l": "લ",
    "m": "મ",
    "n": "ન",
    "o": "ઓ",
    "p": "પ",
    "q": "ક",
    "r": "ર",
    "s": "સ",
    "t": "ત",
    "u": "ઉ",
    "v": "વ",
    "w": "વ",
    "x": "ક્સ",
    "y": "ય",
    "z": "ઝ",
    "0": "૦",
    "1": "૧",
    "2": "૨",
    "3": "૩",
    "4": "૪",
    "5": "૫",
    "6": "૬",
    "7": "૭",
    "8": "૮",
    "9": "૯",
}


@dataclass
class Task2Entry:
    entry_type: str
    source_text: str
    alt_source_text: str
    source_lang: str
    unified_ipa: str
    gujarati_translation: str
    category: str


@dataclass
class TranslationCandidate:
    normalized_source: str
    source_tokens: list[str]
    translation: str
    entry_type: str


class Task2Corpus:
    def __init__(self, corpus_path: str | Path) -> None:
        self.entries: list[Task2Entry] = []
        self.source_to_entry: dict[str, Task2Entry] = {}
        self.phrase_map: dict[str, str] = {}
        self.token_map: dict[str, str] = {}
        self.translation_candidates: list[TranslationCandidate] = []
        self._load(Path(corpus_path))

    def _register_phrase(self, text: str, translation: str, entry_type: str) -> None:
        normalized = normalize_text(text)
        if not normalized or not translation:
            return
        self.phrase_map[normalized] = translation
        self.translation_candidates.append(
            TranslationCandidate(
                normalized_source=normalized,
                source_tokens=word_tokens(text),
                translation=translation,
                entry_type=entry_type,
            )
        )

    def _load(self, path: Path) -> None:
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                entry = Task2Entry(
                    entry_type=row["entry_type"],
                    source_text=row["source_text"],
                    alt_source_text=row["alt_source_text"],
                    source_lang=row["source_lang"],
                    unified_ipa=row["unified_ipa"],
                    gujarati_translation=row["gujarati_translation"],
                    category=row["category"],
                )
                self.entries.append(entry)
                for key in filter(None, [row["source_text"], row["alt_source_text"]]):
                    self.source_to_entry[normalize_text(key)] = entry
                if entry.entry_type in {"phrase", "segment", "term"}:
                    self._register_phrase(entry.source_text, entry.gujarati_translation, entry.entry_type)
                    if entry.alt_source_text:
                        self._register_phrase(entry.alt_source_text, entry.gujarati_translation, entry.entry_type)
                if entry.entry_type == "token":
                    key = normalize_text(entry.source_text)
                    if key and entry.gujarati_translation:
                        self.token_map[key] = entry.gujarati_translation

    def lookup(self, text: str) -> Task2Entry | None:
        return self.source_to_entry.get(normalize_text(text))


class UnifiedIPAMapper:
    def __init__(self, corpus_path: str | Path) -> None:
        self.corpus = Task2Corpus(corpus_path)

    def text_to_ipa(self, text: str) -> str:
        ipa_tokens = []
        for token in tokenize(text):
            entry = self.corpus.lookup(token)
            if entry and entry.unified_ipa:
                ipa_tokens.append(entry.unified_ipa)
            elif re.fullmatch(r"[\u0900-\u097F]+", token):
                ipa_tokens.append(self._deva_to_ipa(token))
            elif re.fullmatch(r"[A-Za-z0-9_']+", token):
                ipa_tokens.append(self._roman_to_ipa(token.lower()))
            else:
                ipa_tokens.append(token)
        return " ".join(ipa_tokens)

    def _deva_to_ipa(self, word: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(word):
            char = word[index]
            if char in DEVANAGARI_VOWELS:
                output.append(DEVANAGARI_VOWELS[char])
                index += 1
                continue
            if char in DEVANAGARI_CONSONANTS:
                base = DEVANAGARI_CONSONANTS[char]
                next_char = word[index + 1] if index + 1 < len(word) else ""
                if next_char in DEVANAGARI_MATRAS:
                    output.append(base + DEVANAGARI_MATRAS[next_char])
                    index += 2
                    continue
                if next_char == "्":
                    output.append(base)
                    index += 2
                    continue
                output.append(base + "ə")
                index += 1
                continue
            if char in {"ं", "ँ"}:
                output.append("̃")
            elif char == "ः":
                output.append("h")
            index += 1
        if output and output[-1].endswith("ə"):
            output[-1] = output[-1][:-1]
        return "".join(output)

    def _roman_to_ipa(self, word: str) -> str:
        transformed = word
        for src, tgt in ROMAN_RULES:
            transformed = transformed.replace(src, f" {tgt} ")
        pieces = []
        for token in transformed.split():
            if all(char in ROMAN_CHARS for char in token):
                pieces.append("".join(ROMAN_CHARS[char] for char in token))
            else:
                pieces.append(token)
        return "".join(pieces)


class CorpusTranslator:
    def __init__(self, corpus_path: str | Path) -> None:
        self.corpus = Task2Corpus(corpus_path)

    def translate(self, text: str) -> str:
        units = self._split_units(text)
        translated_units = [self._translate_unit(unit) for unit in units if unit.strip()]
        translated_units = [unit for unit in translated_units if unit.strip()]
        return "\n\n".join(self._final_gujarati_cleanup(unit) for unit in translated_units)

    def _split_units(self, text: str) -> list[str]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        units: list[str] = []
        for block in blocks:
            parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(block) if part.strip()]
            if parts:
                units.extend(parts)
            else:
                units.append(block)
        return units or [text.strip()]

    def _translate_unit(self, unit: str) -> str:
        normalized_unit = normalize_text(unit)
        if not normalized_unit:
            return ""
        if normalized_unit in self.corpus.phrase_map:
            return self.corpus.phrase_map[normalized_unit]

        fuzzy_translation, fuzzy_score = self._best_fuzzy_translation(unit)
        unit_length = len(word_tokens(unit))
        whole_unit_threshold = 0.52 if unit_length >= 20 else 0.58 if unit_length >= 8 else 0.70
        if fuzzy_translation is not None and fuzzy_score >= whole_unit_threshold:
            return fuzzy_translation

        tokens = tokenize(unit)
        translated: list[str] = []
        index = 0
        max_span = 10
        while index < len(tokens):
            best_translation = None
            best_span = 0
            for span in range(min(max_span, len(tokens) - index), 0, -1):
                phrase = normalize_text(" ".join(tokens[index : index + span]))
                if phrase in self.corpus.phrase_map:
                    best_translation = self.corpus.phrase_map[phrase]
                    best_span = span
                    break
            if best_translation is not None:
                translated.append(best_translation)
                index += best_span
                continue

            fuzzy_translation, fuzzy_span = self._best_fuzzy_span(tokens, index)
            if fuzzy_translation is not None:
                translated.append(fuzzy_translation)
                index += fuzzy_span
                continue

            token = tokens[index]
            key = normalize_text(token)
            if key in self.corpus.token_map:
                translated.append(self.corpus.token_map[key])
            elif re.fullmatch(r"[\u0900-\u097F]+", token):
                translated.append(self._deva_to_gujarati(token))
            elif re.fullmatch(r"[A-Za-z0-9_']+", token):
                translated.append(self._roman_to_gujarati(token))
            else:
                translated.append(token)
            index += 1
        return self._detokenize(translated)

    def _score_fuzzy_translation(self, unit_tokens: list[str], normalized_unit: str) -> tuple[str | None, float]:
        if not unit_tokens:
            return None, 0.0
        counter = Counter(unit_tokens)
        best_score = 0.0
        best_translation: str | None = None

        for candidate in self.corpus.translation_candidates:
            if len(unit_tokens) > 6 and candidate.entry_type == "term":
                continue
            if len(unit_tokens) <= 3 and candidate.entry_type == "segment":
                continue

            overlap = sum((counter & Counter(candidate.source_tokens)).values())
            if overlap == 0:
                continue

            precision = overlap / max(len(unit_tokens), 1)
            recall = overlap / max(len(candidate.source_tokens), 1)
            token_f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
            char_ratio = SequenceMatcher(None, normalized_unit, candidate.normalized_source).ratio()
            length_ratio = min(len(unit_tokens), len(candidate.source_tokens)) / max(len(unit_tokens), len(candidate.source_tokens), 1)
            type_bonus = 0.05 if candidate.entry_type == "segment" else 0.02 if candidate.entry_type == "phrase" else 0.0
            score = (0.55 * token_f1) + (0.30 * char_ratio) + (0.15 * length_ratio) + type_bonus
            if score > best_score:
                best_score = score
                best_translation = candidate.translation

        return best_translation, best_score

    def _best_fuzzy_translation(self, unit: str) -> tuple[str | None, float]:
        unit_tokens = word_tokens(unit)
        normalized_unit = normalize_text(unit)
        return self._score_fuzzy_translation(unit_tokens, normalized_unit)

    def _best_fuzzy_span(self, tokens: list[str], start_index: int) -> tuple[str | None, int]:
        best_translation: str | None = None
        best_span = 0
        best_score = 0.0
        max_candidate_span = min(30, len(tokens) - start_index)
        min_candidate_span = 5
        for span in range(max_candidate_span, min_candidate_span - 1, -1):
            span_text = " ".join(tokens[start_index : start_index + span])
            unit_tokens = word_tokens(span_text)
            normalized_unit = normalize_text(span_text)
            translation, score = self._score_fuzzy_translation(unit_tokens, normalized_unit)
            if translation is not None and score > best_score:
                best_translation = translation
                best_span = span
                best_score = score
        threshold = 0.55 if len(word_tokens(" ".join(tokens[start_index : start_index + max(best_span, 1)]))) >= 10 else 0.60
        return (best_translation, best_span) if best_score >= threshold else (None, 0)

    def _deva_to_gujarati(self, word: str) -> str:
        return "".join(DEVANAGARI_TO_GUJARATI.get(char, char) for char in word)

    def _roman_to_gujarati(self, word: str) -> str:
        lowered = word.lower()
        output: list[str] = []
        index = 0
        while index < len(lowered):
            matched = False
            for src, tgt in ROMAN_TO_GUJARATI_RULES:
                if lowered.startswith(src, index):
                    output.append(tgt)
                    index += len(src)
                    matched = True
                    break
            if matched:
                continue
            char = lowered[index]
            output.append(ROMAN_TO_GUJARATI_CHARS.get(char, char))
            index += 1
        return "".join(output)

    def _detokenize(self, tokens: list[str]) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?।])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]}])", r"\1", text)
        return " ".join(text.split())

    def _final_gujarati_cleanup(self, text: str) -> str:
        cleaned: list[str] = []
        for token in tokenize(text):
            if re.fullmatch(r"[\u0900-\u097F]+", token):
                cleaned.append(self._deva_to_gujarati(token))
            elif re.fullmatch(r"[A-Za-z0-9_']+", token):
                cleaned.append(self._roman_to_gujarati(token))
            else:
                cleaned.append(token)
        return self._detokenize(cleaned)
