from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from assignment2.models.lid import FrameLevelLIDNet, lid_loss
from assignment2.modules.features import log_mel_spectrogram
from assignment2.utils.audio import load_audio

LANG_TO_ID = {"hindi": 0, "english": 1}


@dataclass
class LIDChunk:
    features: torch.Tensor
    language_targets: torch.Tensor
    switch_targets: torch.Tensor
    switch_eval_targets: torch.Tensor


class LIDSequenceDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        chunk_seconds: float,
        switch_boundary_radius_frames: int = 0,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.chunk_seconds = chunk_seconds
        self.switch_boundary_radius_frames = max(int(switch_boundary_radius_frames), 0)
        self.examples = self._build_examples(Path(manifest_path))

    def _build_examples(self, manifest_path: Path) -> list[LIDChunk]:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        with manifest_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                grouped[row["audio_path"]].append(row)

        examples: list[LIDChunk] = []
        chunk_frames = int(round(self.chunk_seconds * self.sample_rate / self.hop_length))

        for audio_path, rows in grouped.items():
            rows = sorted(rows, key=lambda item: float(item["start_sec"]))
            waveform, _ = load_audio(audio_path, target_sr=self.sample_rate, mono=True)
            features = log_mel_spectrogram(
                waveform,
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            ).squeeze(0)
            features = _cmvn(features)

            language_targets = torch.full((features.size(0),), -100, dtype=torch.long)
            switch_targets = torch.zeros(features.size(0), dtype=torch.long)
            switch_eval_targets = torch.zeros(features.size(0), dtype=torch.long)

            for row_idx, row in enumerate(rows):
                start = int(float(row["start_sec"]) * self.sample_rate / self.hop_length)
                end = int(float(row["end_sec"]) * self.sample_rate / self.hop_length)
                end = min(end, features.size(0))
                language_targets[start:end] = LANG_TO_ID[row["lang"].strip().lower()]
                if row_idx > 0 and start < switch_targets.numel():
                    switch_eval_targets[start] = 1
                    boundary_start = max(0, start - self.switch_boundary_radius_frames)
                    boundary_end = min(switch_targets.numel(), start + self.switch_boundary_radius_frames + 1)
                    switch_targets[boundary_start:boundary_end] = 1

            for start in range(0, features.size(0), chunk_frames):
                end = min(start + chunk_frames, features.size(0))
                chunk_features = features[start:end]
                chunk_language = language_targets[start:end]
                chunk_switch = switch_targets[start:end]
                chunk_switch_eval = switch_eval_targets[start:end]
                if (chunk_language >= 0).any():
                    examples.append(LIDChunk(chunk_features, chunk_language, chunk_switch, chunk_switch_eval))
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LIDChunk:
        return self.examples[index]


def collate_lid(batch: list[LIDChunk]) -> dict[str, torch.Tensor]:
    max_frames = max(item.features.size(0) for item in batch)
    feat_dim = batch[0].features.size(-1)
    features = torch.zeros(len(batch), max_frames, feat_dim)
    language_targets = torch.full((len(batch), max_frames), -100, dtype=torch.long)
    switch_targets = torch.full((len(batch), max_frames), -1, dtype=torch.long)
    switch_eval_targets = torch.full((len(batch), max_frames), -1, dtype=torch.long)

    for idx, item in enumerate(batch):
        length = item.features.size(0)
        features[idx, :length] = item.features
        language_targets[idx, :length] = item.language_targets
        switch_targets[idx, :length] = item.switch_targets
        switch_eval_targets[idx, :length] = item.switch_eval_targets

    return {
        "features": features,
        "language_targets": language_targets,
        "switch_targets": switch_targets,
        "switch_eval_targets": switch_eval_targets,
    }


def _cmvn(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(eps)
    return (features - mean) / std


def _time_mask(features: torch.Tensor, max_width: int = 16) -> torch.Tensor:
    if features.size(0) < 8:
        return features
    width = min(max_width, max(features.size(0) // 10, 1))
    if width <= 0 or features.size(0) <= width:
        return features
    start = int(torch.randint(0, features.size(0) - width + 1, (1,)).item())
    augmented = features.clone()
    augmented[start : start + width] = 0.0
    return augmented


def _moving_average(sequence: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1 or sequence.numel() == 0:
        return sequence
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    weights = torch.ones((1, 1, kernel_size), dtype=sequence.dtype, device=sequence.device) / kernel_size
    smoothed = F.conv1d(
        F.pad(sequence.view(1, 1, -1), (padding, padding), mode="replicate"),
        weights,
    )
    return smoothed.view(-1)


def decode_language_sequence(
    language_probs: torch.Tensor,
    english_enter_threshold: float = 0.55,
    english_exit_threshold: float = 0.18,
    smoothing_frames: int = 11,
    minimum_english_frames: int = 5,
) -> torch.Tensor:
    if language_probs.ndim != 2 or language_probs.size(-1) != 2:
        return language_probs.argmax(dim=-1)

    english_prob = _moving_average(language_probs[:, 1], kernel_size=smoothing_frames)
    prediction = torch.zeros(english_prob.size(0), dtype=torch.long, device=language_probs.device)
    state = 1 if float(english_prob[0]) >= english_enter_threshold else 0
    prediction[0] = state

    for frame_idx in range(1, english_prob.size(0)):
        probability = float(english_prob[frame_idx])
        if state == 0 and probability >= english_enter_threshold:
            state = 1
        elif state == 1 and probability <= english_exit_threshold:
            state = 0
        prediction[frame_idx] = state

    if minimum_english_frames > 1:
        start = None
        for frame_idx, class_id in enumerate(prediction.tolist() + [0]):
            if class_id == 1 and start is None:
                start = frame_idx
            elif class_id == 0 and start is not None:
                if frame_idx - start < minimum_english_frames:
                    prediction[start:frame_idx] = 0
                start = None
    return prediction


def decode_switch_sequence(
    language_pred: torch.Tensor,
    switch_probs: torch.Tensor,
    switch_probability_threshold: float = 0.35,
    min_separation_frames: int = 6,
) -> torch.Tensor:
    switch_pred = torch.zeros_like(language_pred)
    if language_pred.numel() == 0:
        return switch_pred

    transition_indices = (language_pred[1:] != language_pred[:-1]).nonzero(as_tuple=False).flatten() + 1
    switch_pred[transition_indices] = 1

    if switch_probs.numel() == language_pred.numel():
        smoothed_switch = _moving_average(switch_probs, kernel_size=5)
        candidate_indices = (smoothed_switch >= switch_probability_threshold).nonzero(as_tuple=False).flatten()
        for index in candidate_indices.tolist():
            left = max(0, index - min_separation_frames)
            right = min(switch_pred.numel(), index + min_separation_frames + 1)
            if int(switch_pred[left:right].sum().item()) == 0:
                switch_pred[index] = 1
    return switch_pred


def train_lid_model(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    chunk_seconds: float,
    model_kwargs: dict[str, int],
    switch_loss_weight: float,
    switch_boundary_radius_frames: int,
    device: str,
) -> FrameLevelLIDNet:
    dataset = LIDSequenceDataset(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        chunk_seconds=chunk_seconds,
        switch_boundary_radius_frames=switch_boundary_radius_frames,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_lid)
    model = FrameLevelLIDNet(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    total_positive = sum(int((example.switch_targets > 0).sum().item()) for example in dataset.examples)
    total_frames = sum(int(example.switch_targets.numel()) for example in dataset.examples)
    total_negative = max(total_frames - total_positive, 1)
    switch_pos_weight = min(total_negative / max(total_positive, 1), 20.0)

    for _ in range(epochs):
        model.train()
        for batch in loader:
            features = batch["features"].clone()
            for item_idx in range(features.size(0)):
                features[item_idx] = _time_mask(features[item_idx])
            features = features.to(device)
            language_targets = batch["language_targets"].to(device)
            switch_targets = batch["switch_targets"].to(device)
            outputs = model(features)
            loss = lid_loss(
                outputs,
                language_targets,
                switch_targets,
                switch_loss_weight=switch_loss_weight,
                switch_pos_weight=switch_pos_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return model


def predict_lid_frames(
    model: FrameLevelLIDNet,
    features: torch.Tensor,
    english_enter_threshold: float = 0.55,
    english_exit_threshold: float = 0.18,
    smoothing_frames: int = 11,
    minimum_english_frames: int = 5,
    switch_probability_threshold: float = 0.35,
    switch_min_separation_frames: int = 6,
) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        outputs = model(features.unsqueeze(0))
        language_probs = outputs["language_logits"].softmax(dim=-1).squeeze(0)
        switch_probs = outputs["switch_logits"].sigmoid().squeeze(0)
        language_pred = decode_language_sequence(
            language_probs,
            english_enter_threshold=english_enter_threshold,
            english_exit_threshold=english_exit_threshold,
            smoothing_frames=smoothing_frames,
            minimum_english_frames=minimum_english_frames,
        )
        switch_pred = decode_switch_sequence(
            language_pred,
            switch_probs,
            switch_probability_threshold=switch_probability_threshold,
            min_separation_frames=switch_min_separation_frames,
        )
    return {
        "language_probs": language_probs,
        "language_pred": language_pred,
        "switch_probs": switch_probs,
        "switch_pred": switch_pred,
    }
