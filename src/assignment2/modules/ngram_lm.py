from __future__ import annotations

import math
import re
from collections import Counter, defaultdict


TERM_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u0900-\u097F]{2,}", re.UNICODE)


class NGramLanguageModel:
    def __init__(self, order: int = 3) -> None:
        self.order = order
        self.ngram_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()
        self.technical_terms: list[str] = []

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.casefold().strip().split())

    def fit_from_token_sequences(self, sequences: list[list[str]]) -> None:
        self.ngram_counts.clear()
        self.vocab.clear()
        for sequence in sequences:
            padded = ["<s>"] * (self.order - 1) + sequence + ["</s>"]
            for index in range(self.order - 1, len(padded)):
                history = tuple(padded[index - self.order + 1 : index])
                token = padded[index]
                self.ngram_counts[history][token] += 1
                self.vocab.add(token)

    def fit_from_text(self, text: str, tokenizer=None) -> None:
        sequences: list[list[str]] = []
        normalized_lines = [self._normalize_text(line) for line in text.splitlines() if line.strip()]
        for line in normalized_lines:
            if tokenizer is None:
                sequence = line.split()
            else:
                sequence = [token for token in tokenizer.tokenize(line) if token.strip()]
            if sequence:
                sequences.append(sequence)
        self.fit_from_token_sequences(sequences)
        extracted_terms: set[str] = set(normalized_lines)
        extracted_terms.update(match.group(0).casefold() for match in TERM_PATTERN.finditer(text))
        self.technical_terms = sorted(extracted_terms, key=len, reverse=True)

    def conditional_log_prob(self, history: tuple[str, ...], token: str) -> float:
        history = tuple(history[-(self.order - 1) :])
        counts = self.ngram_counts.get(history, Counter())
        numerator = counts[token] + 1
        denominator = sum(counts.values()) + max(len(self.vocab), 1)
        return float(math.log(numerator / denominator))

    def prefix_bonus(self, decoded_text: str, term_bonus: float = 1.5, partial_bonus_scale: float = 0.35) -> float:
        text = self._normalize_text(decoded_text)
        if not text:
            return 0.0
        best_bonus = 0.0
        for term in self.technical_terms:
            if term and term in text:
                return term_bonus
            max_overlap = min(len(term), len(text))
            for overlap in range(max_overlap, 2, -1):
                if term.startswith(text[-overlap:]):
                    best_bonus = max(best_bonus, partial_bonus_scale * term_bonus * (overlap / len(term)))
                    break
        return best_bonus
