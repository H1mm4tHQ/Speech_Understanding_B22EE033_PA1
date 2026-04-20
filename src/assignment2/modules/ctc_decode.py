from __future__ import annotations

from dataclasses import dataclass

import torch

from assignment2.modules.ngram_lm import NGramLanguageModel


@dataclass
class Beam:
    token_ids: tuple[int, ...]
    score: float


class ConstrainedCTCBeamSearch:
    """CTC beam search with N-gram bias and technical-term bonus."""

    def __init__(
        self,
        beam_size: int,
        lm_weight: float,
        length_penalty: float,
        term_bonus: float,
        blank_id: int,
        ngram_lm: NGramLanguageModel,
        id_to_token: dict[int, str],
    ) -> None:
        self.beam_size = beam_size
        self.lm_weight = lm_weight
        self.length_penalty = length_penalty
        self.term_bonus = term_bonus
        self.blank_id = blank_id
        self.ngram_lm = ngram_lm
        self.id_to_token = id_to_token

    def _collapse_ctc(self, token_ids: tuple[int, ...]) -> tuple[int, ...]:
        collapsed: list[int] = []
        prev = None
        for token_id in token_ids:
            if token_id == self.blank_id:
                prev = token_id
                continue
            if token_id != prev:
                collapsed.append(token_id)
            prev = token_id
        return tuple(collapsed)

    def _tokens_to_text(self, token_ids: tuple[int, ...]) -> str:
        pieces = [self.id_to_token[token_id] for token_id in self._collapse_ctc(token_ids)]
        text = "".join(pieces).replace("|", " ").replace("▁", " ")
        return " ".join(text.strip().split())

    def decode(self, logits: torch.Tensor) -> tuple[str, list[int]]:
        log_probs = logits.log_softmax(dim=-1)
        beams = [Beam(token_ids=tuple(), score=0.0)]

        for frame in log_probs:
            topk_scores, topk_ids = torch.topk(frame, k=min(self.beam_size * 6, frame.numel()))
            candidates: dict[tuple[int, ...], float] = {}

            for beam in beams:
                blank_score = beam.score + float(frame[self.blank_id])
                candidates[beam.token_ids] = max(candidates.get(beam.token_ids, float("-inf")), blank_score)

                collapsed_history = self._collapse_ctc(beam.token_ids)
                history = tuple(self.id_to_token[token_id] for token_id in collapsed_history[-(self.ngram_lm.order - 1) :])

                for token_score, token_id in zip(topk_scores.tolist(), topk_ids.tolist()):
                    if token_id == self.blank_id:
                        continue
                    next_token = self.id_to_token[token_id]
                    new_tokens = beam.token_ids + (token_id,)
                    partial_text = self._tokens_to_text(new_tokens)
                    lm_score = self.ngram_lm.conditional_log_prob(history, next_token)
                    bonus = self.ngram_lm.prefix_bonus(partial_text, term_bonus=self.term_bonus)
                    score = beam.score + token_score + self.lm_weight * lm_score + bonus
                    score -= self.length_penalty * len(self._collapse_ctc(new_tokens))
                    candidates[new_tokens] = max(candidates.get(new_tokens, float("-inf")), score)

            beams = [
                Beam(token_ids=tokens, score=score)
                for tokens, score in sorted(candidates.items(), key=lambda item: item[1], reverse=True)[: self.beam_size]
            ]

        best = beams[0]
        final_tokens = list(self._collapse_ctc(best.token_ids))
        final_text = self._tokens_to_text(best.token_ids)
        return final_text.strip(), final_tokens
