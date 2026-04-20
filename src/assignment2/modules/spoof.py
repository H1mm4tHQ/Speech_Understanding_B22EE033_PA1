from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch.utils.data import DataLoader, Dataset

from assignment2.modules.evaluation import equal_error_rate, threshold_at_equal_error_rate
from assignment2.modules.features import lfcc_features
from assignment2.utils.audio import chunk_audio, frame_rms, load_audio, slice_audio


LABEL_MAP = {"bona_fide": 0, "spoof": 1}


@dataclass
class SpoofExample:
    features: torch.Tensor
    label: int


@dataclass
class SpoofRecord:
    audio_path: str
    label: str
    start_sec: float
    end_sec: float
    split: str
    group_id: str
    segment_id: str


class SpoofDataset(Dataset):
    def __init__(self, manifest_path: str | Path, sample_rate: int, n_lfcc: int, split: str | None = None) -> None:
        self.examples: list[SpoofExample] = []
        with Path(manifest_path).open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if split and row.get("split", "").strip().lower() != split:
                    continue
                waveform, _ = load_audio(row["audio_path"], target_sr=sample_rate, mono=True)
                if row.get("start_sec") and row.get("end_sec"):
                    waveform = slice_audio(
                        waveform,
                        sample_rate=sample_rate,
                        start_sec=float(row["start_sec"]),
                        end_sec=float(row["end_sec"]),
                    )
                feats = lfcc_features(waveform, sample_rate=sample_rate, n_lfcc=n_lfcc).squeeze(0)
                label = LABEL_MAP[row["label"].strip().lower()]
                self.examples.append(SpoofExample(feats, label))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SpoofExample:
        return self.examples[index]


def collate_spoof(batch: list[SpoofExample]) -> tuple[torch.Tensor, torch.Tensor]:
    max_frames = max(item.features.size(0) for item in batch)
    feat_dim = batch[0].features.size(-1)
    feats = torch.zeros(len(batch), 1, max_frames, feat_dim)
    labels = torch.tensor([item.label for item in batch], dtype=torch.long)
    for idx, item in enumerate(batch):
        length = item.features.size(0)
        feats[idx, 0, :length] = item.features
    return feats, labels


class AntiSpoofCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.net(features)
        x = x.flatten(1)
        return self.classifier(x)


def _choose_evenly_spaced(records: list[tuple[float, float]], target_count: int) -> list[tuple[float, float]]:
    if target_count <= 0:
        return []
    if len(records) <= target_count:
        return records
    positions = torch.linspace(0, len(records) - 1, steps=target_count).round().long().tolist()
    return [records[idx] for idx in positions]


def build_spoof_segment_manifest(
    bona_fide_audio: str | Path,
    spoof_audio: str | Path,
    output_manifest: str | Path,
    sample_rate: int,
    chunk_seconds: float = 4.0,
    min_rms: float = 0.005,
    group_chunks: int = 2,
    max_chunks_per_label: int = 60,
) -> dict[str, int]:
    audio_specs = [
        ("bona_fide", str(bona_fide_audio), "bf"),
        ("spoof", str(spoof_audio), "sp"),
    ]
    selected_by_label: dict[str, list[tuple[float, float]]] = {}
    source_paths: dict[str, str] = {}

    for label, audio_path, _ in audio_specs:
        waveform, sr = load_audio(audio_path, target_sr=sample_rate, mono=True)
        source_paths[label] = audio_path
        candidate_segments: list[tuple[float, float]] = []
        for start_sec, end_sec, chunk in chunk_audio(waveform, sample_rate=sr, chunk_seconds=chunk_seconds, overlap_seconds=0.0):
            rms = float(frame_rms(chunk, frame_length=min(400, chunk.size(-1)), hop_length=max(160, min(400, chunk.size(-1)))).mean().item())
            if rms >= min_rms and (end_sec - start_sec) >= (0.8 * chunk_seconds):
                candidate_segments.append((start_sec, end_sec))
        selected_by_label[label] = candidate_segments

    target_count = min(min(len(records) for records in selected_by_label.values()), max_chunks_per_label)
    rows: list[SpoofRecord] = []
    for label, _, prefix in audio_specs:
        chosen = _choose_evenly_spaced(selected_by_label[label], target_count)
        for idx, (start_sec, end_sec) in enumerate(chosen):
            group_index = idx // max(group_chunks, 1)
            remainder = group_index % 5
            if remainder == 0:
                split = "test"
            elif remainder == 1:
                split = "val"
            else:
                split = "train"
            rows.append(
                SpoofRecord(
                    audio_path=source_paths[label],
                    label=label,
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    split=split,
                    group_id=f"{prefix}_g{group_index:03d}",
                    segment_id=f"{prefix}_{idx:03d}",
                )
            )

    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "label", "start_sec", "end_sec", "split", "group_id", "segment_id"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    return {
        "balanced_chunks_per_label": target_count,
        "total_rows": len(rows),
        "train_rows": sum(row.split == "train" for row in rows),
        "val_rows": sum(row.split == "val" for row in rows),
        "test_rows": sum(row.split == "test" for row in rows),
    }


def _make_loader(dataset: SpoofDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_spoof)


def _run_epoch(model: AntiSpoofCNN, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: str) -> float:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_items = 0
    for feats, labels in loader:
        feats = feats.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(feats)
            loss = F.cross_entropy(logits, labels)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def _collect_scores(model: AntiSpoofCNN, loader: DataLoader, device: str) -> tuple[list[int], list[float]]:
    model.eval()
    labels_all: list[int] = []
    scores_all: list[float] = []
    with torch.inference_mode():
        for feats, labels in loader:
            feats = feats.to(device)
            logits = model(feats)
            probs = logits.softmax(dim=-1)[:, 1].cpu().tolist()
            scores_all.extend(float(score) for score in probs)
            labels_all.extend(int(label) for label in labels.tolist())
    return labels_all, scores_all


def _save_roc_plot(labels: list[int], scores: list[float], output_path: str | Path) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Anti-Spoof ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return roc_auc


def _save_confusion_plot(matrix: list[list[int]], output_path: str | Path) -> None:
    plt.figure(figsize=(4.5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["bona_fide", "spoof"])
    plt.yticks([0, 1], ["bona_fide", "spoof"])
    plt.xlabel("Predicted")
    plt.ylabel("Reference")
    plt.title("Anti-Spoof Confusion Matrix")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            plt.text(col_idx, row_idx, str(value), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def evaluate_spoof_experiment(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    sample_rate: int,
    n_lfcc: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: str,
    roc_plot_path: str | Path,
    confusion_plot_path: str | Path,
) -> dict[str, object]:
    train_dataset = SpoofDataset(manifest_path, sample_rate=sample_rate, n_lfcc=n_lfcc, split="train")
    val_dataset = SpoofDataset(manifest_path, sample_rate=sample_rate, n_lfcc=n_lfcc, split="val")
    test_dataset = SpoofDataset(manifest_path, sample_rate=sample_rate, n_lfcc=n_lfcc, split="test")
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Spoof manifest must contain non-empty train, val, and test splits.")

    train_loader = _make_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = _make_loader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AntiSpoofCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(epochs):
        _run_epoch(model, train_loader, optimizer, device)
        val_loss = _run_epoch(model, val_loader, None, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    val_labels, val_scores = _collect_scores(model, val_loader, device)
    test_labels, test_scores = _collect_scores(model, test_loader, device)
    target_scores = [score for label, score in zip(val_labels, val_scores) if label == LABEL_MAP["spoof"]]
    nontarget_scores = [score for label, score in zip(val_labels, val_scores) if label == LABEL_MAP["bona_fide"]]
    threshold, val_eer = threshold_at_equal_error_rate(target_scores, nontarget_scores)

    predictions = [1 if score >= threshold else 0 for score in test_scores]
    matrix = confusion_matrix(test_labels, predictions, labels=[0, 1]).tolist()
    accuracy = sum(int(pred == label) for pred, label in zip(predictions, test_labels)) / max(len(test_labels), 1)
    roc_auc = _save_roc_plot(test_labels, test_scores, roc_plot_path)
    _save_confusion_plot(matrix, confusion_plot_path)

    test_target_scores = [score for label, score in zip(test_labels, test_scores) if label == LABEL_MAP["spoof"]]
    test_nontarget_scores = [score for label, score in zip(test_labels, test_scores) if label == LABEL_MAP["bona_fide"]]
    test_eer = equal_error_rate(test_target_scores, test_nontarget_scores)

    return {
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "test_examples": len(test_dataset),
        "threshold_from_val_eer": float(threshold),
        "val_eer": float(val_eer),
        "test_eer": float(test_eer),
        "test_accuracy": float(accuracy),
        "test_roc_auc": float(roc_auc),
        "test_confusion_matrix": matrix,
    }


def train_spoof_model(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    sample_rate: int,
    n_lfcc: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: str,
) -> AntiSpoofCNN:
    dataset = SpoofDataset(manifest_path, sample_rate=sample_rate, n_lfcc=n_lfcc)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_spoof)
    model = AntiSpoofCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        _run_epoch(model, loader, optimizer, device)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return model
