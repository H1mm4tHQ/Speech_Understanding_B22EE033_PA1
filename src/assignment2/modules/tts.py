from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoTokenizer, VitsModel

from assignment2.modules.prosody import apply_prosody_warp, extract_prosody, warp_prosody
from assignment2.utils.audio import load_audio, save_audio


class GujaratiTTS:
    """VITS wrapper with DTW-based post synthesis prosody warping."""

    def __init__(self, model_name: str, sample_rate: int, speaker_id: int = 0) -> None:
        self.output_sample_rate = int(sample_rate)
        self.sample_rate = int(sample_rate)
        self.speaker_id = speaker_id
        model_source, local_only = self._resolve_model_source(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
        self.model = VitsModel.from_pretrained(model_source, local_files_only=local_only)
        self.model.eval()
        if hasattr(self.model.config, "sampling_rate"):
            self.sample_rate = int(self.model.config.sampling_rate)

    def _resolve_model_source(self, model_name: str) -> tuple[str | Path, bool]:
        if Path(model_name).exists():
            return Path(model_name), True
        repo_dir_name = f"models--{model_name.replace('/', '--')}"
        cache_roots = [
            Path.cwd() / ".hf_cache" / "hub",
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
        for cache_root in cache_roots:
            repo_dir = cache_root / repo_dir_name / "snapshots"
            if not repo_dir.exists():
                continue
            for snapshot_dir in sorted(repo_dir.iterdir(), reverse=True):
                required = ["config.json", "tokenizer_config.json", "vocab.json", "model.safetensors"]
                if all((snapshot_dir / name).exists() for name in required):
                    return snapshot_dir, True
        return model_name, False

    def _split_text(self, text: str) -> list[str]:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text).strip()
        raw_chunks = [part.strip() for part in re.split(r"[।.!?\n]+", text) if part.strip()]
        segments: list[str] = []
        for chunk in raw_chunks:
            comma_split = [part.strip(" ,;:") for part in re.split(r"[,;:]+", chunk) if part.strip(" ,;:")]
            for part in comma_split:
                words = part.split()
                if len(words) <= 20:
                    segments.append(part)
                    continue
                for start in range(0, len(words), 18):
                    segment = " ".join(words[start : start + 18]).strip()
                    if segment:
                        segments.append(segment)
        if not segments:
            return [text.strip()]

        merged_segments: list[str] = []
        for segment in segments:
            words = segment.split()
            if merged_segments and len(words) <= 4:
                merged_segments[-1] = f"{merged_segments[-1]} {segment}".strip()
            else:
                merged_segments.append(segment)
        return merged_segments

    def synthesize(self, text: str, speaking_rate: float = 1.0, add_pauses: bool = True) -> torch.Tensor:
        waveform_chunks: list[torch.Tensor] = []
        for segment in self._split_text(text):
            if not segment.strip():
                continue
            inputs = self.tokenizer(segment, return_tensors="pt")
            if "input_ids" not in inputs or inputs["input_ids"].numel() == 0 or inputs["input_ids"].shape[-1] == 0:
                continue
            kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask"),
                "speaking_rate": speaking_rate,
            }
            if self.speaker_id is not None:
                kwargs["speaker_id"] = self.speaker_id
            with torch.inference_mode():
                output = self.model(**kwargs)
            waveform = output.waveform
            waveform = waveform if waveform.dim() == 2 else waveform.unsqueeze(0)
            waveform_chunks.append(waveform)
            if add_pauses:
                silence = torch.zeros((1, int(0.15 * self.sample_rate)), dtype=waveform.dtype)
                waveform_chunks.append(silence)
        return torch.cat(waveform_chunks, dim=-1) if waveform_chunks else torch.zeros(1, 1)

    def _match_duration(self, waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
        target_samples = max(int(target_samples), 1)
        if waveform.size(-1) == target_samples:
            return waveform
        if waveform.numel() == 0:
            return torch.zeros((1, target_samples), dtype=torch.float32)
        current_samples = int(waveform.size(-1))
        ratio = target_samples / max(current_samples, 1)
        if 0.9 <= ratio <= 1.1:
            return F.interpolate(
                waveform.unsqueeze(0),
                size=target_samples,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
        if target_samples > current_samples:
            padding = torch.zeros((waveform.size(0), target_samples - current_samples), dtype=waveform.dtype)
            return torch.cat([waveform, padding], dim=-1)
        return waveform[..., :target_samples]

    def _allocate_target_lengths(self, total_frames: int, base_waveforms: list[torch.Tensor]) -> list[int]:
        total_frames = max(int(total_frames), len(base_waveforms))
        base_lengths = [max(int(waveform.size(-1)), 1) for waveform in base_waveforms]
        total_base = max(sum(base_lengths), 1)
        raw_allocations = [(length / total_base) * total_frames for length in base_lengths]
        target_lengths = [max(int(round(value)), 1) for value in raw_allocations]

        diff = total_frames - sum(target_lengths)
        if diff != 0 and target_lengths:
            order = sorted(
                range(len(target_lengths)),
                key=lambda idx: raw_allocations[idx] - target_lengths[idx],
                reverse=(diff > 0),
            )
            order_index = 0
            while diff != 0 and order_index < (len(order) * 4 + abs(diff)):
                idx = order[order_index % len(order)]
                if diff > 0:
                    target_lengths[idx] += 1
                    diff -= 1
                elif target_lengths[idx] > 1:
                    target_lengths[idx] -= 1
                    diff += 1
                order_index += 1

        if sum(target_lengths) != total_frames and target_lengths:
            target_lengths[-1] += total_frames - sum(target_lengths)
        return target_lengths

    def _maybe_resample_output(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.output_sample_rate == self.sample_rate:
            return waveform
        return torchaudio.functional.resample(waveform, self.sample_rate, self.output_sample_rate)

    def synthesize_with_prosody(
        self,
        text: str,
        reference_professor_audio: str | Path,
        output_path: str | Path,
        flat_output_path: str | Path | None,
        speaking_rate: float,
        hop_ms: float,
        frame_length: int,
        hop_length: int,
    ) -> torch.Tensor:
        professor_waveform, professor_sr = load_audio(reference_professor_audio, target_sr=self.sample_rate, mono=True)
        text_segments = [segment for segment in self._split_text(text) if segment.strip()]
        if not text_segments:
            empty = torch.zeros(1, 1)
            if flat_output_path is not None:
                save_audio(flat_output_path, empty, self.output_sample_rate)
            save_audio(output_path, empty, self.output_sample_rate)
            return empty

        base_waveforms = [self.synthesize(segment, speaking_rate=speaking_rate, add_pauses=False) for segment in text_segments]
        target_lengths = self._allocate_target_lengths(professor_waveform.size(-1), base_waveforms)
        professor_segments: list[torch.Tensor] = []
        start = 0
        for idx, target_length in enumerate(target_lengths):
            end = start + target_length
            if idx == len(target_lengths) - 1:
                end = professor_waveform.size(-1)
            professor_segments.append(professor_waveform[..., start:end])
            start = end

        waveform_chunks: list[torch.Tensor] = []
        flat_waveform_chunks: list[torch.Tensor] = []

        for idx, segment in enumerate(text_segments):
            base_waveform = base_waveforms[idx]
            professor_chunk = professor_segments[min(idx, len(professor_segments) - 1)]
            target_samples = max(int(professor_chunk.size(-1)), 1)
            flat_waveform = self._match_duration(base_waveform, target_samples)
            flat_waveform_chunks.append(flat_waveform)

            source_profile = extract_prosody(
                professor_chunk,
                sample_rate=professor_sr,
                hop_ms=hop_ms,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            target_profile = extract_prosody(
                base_waveform,
                sample_rate=self.sample_rate,
                hop_ms=hop_ms,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            warped_profile = warp_prosody(source_profile, target_profile)
            warped_waveform = apply_prosody_warp(
                base_waveform,
                sample_rate=self.sample_rate,
                original_target=target_profile,
                warped_target=warped_profile,
                hop_length=hop_length,
            )
            waveform_chunks.append(self._match_duration(warped_waveform, target_samples))

        flat_waveform = torch.cat(flat_waveform_chunks, dim=-1) if flat_waveform_chunks else torch.zeros(1, 1)
        final_waveform = torch.cat(waveform_chunks, dim=-1) if waveform_chunks else torch.zeros(1, 1)
        flat_waveform = self._maybe_resample_output(flat_waveform)
        final_waveform = self._maybe_resample_output(final_waveform)
        if flat_output_path is not None:
            save_audio(flat_output_path, flat_waveform, self.output_sample_rate)
        save_audio(output_path, final_waveform, self.output_sample_rate)
        return final_waveform
