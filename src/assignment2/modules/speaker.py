from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from assignment2.modules.features import mfcc_features
from assignment2.utils.audio import load_audio


class TDNNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.activation(self.conv(x)))


class StatisticsPooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)
        return torch.cat([mean, std], dim=-1)


class XVectorEncoder(nn.Module):
    def __init__(self, input_dim: int = 40, embedding_dim: int = 256) -> None:
        super().__init__()
        self.tdnn1 = TDNNBlock(input_dim, 512, kernel_size=5, dilation=1)
        self.tdnn2 = TDNNBlock(512, 512, kernel_size=3, dilation=2)
        self.tdnn3 = TDNNBlock(512, 512, kernel_size=3, dilation=3)
        self.tdnn4 = TDNNBlock(512, 512, kernel_size=1, dilation=1)
        self.tdnn5 = TDNNBlock(512, 1500, kernel_size=1, dilation=1)
        self.pool = StatisticsPooling()
        self.segment6 = nn.Linear(3000, 512)
        self.segment7 = nn.Linear(512, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = self.pool(x)
        x = F.relu(self.segment6(x))
        x = self.segment7(x)
        return F.normalize(x, dim=-1)


class SpeakerEmbeddingExtractor:
    def __init__(self, checkpoint_path: str | Path | None, sample_rate: int, embedding_dim: int = 256) -> None:
        self.sample_rate = sample_rate
        self.model = XVectorEncoder(embedding_dim=embedding_dim)
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.eval()

    def extract(self, audio_path: str | Path) -> torch.Tensor:
        waveform, _ = load_audio(audio_path, target_sr=self.sample_rate, mono=True)
        feats = mfcc_features(waveform, sample_rate=self.sample_rate).squeeze(0)
        with torch.inference_mode():
            embedding = self.model(feats.unsqueeze(0)).squeeze(0)
        return embedding

