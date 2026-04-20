from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameLevelLIDNet(nn.Module):
    """Multi-head frame-level LID with an auxiliary switch-boundary head."""

    def __init__(
        self,
        input_dim: int = 80,
        conv_dim: int = 128,
        hidden_dim: int = 256,
        num_languages: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_dim),
            nn.GELU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_dim),
            nn.GELU(),
        )
        self.encoder = nn.LSTM(
            input_size=conv_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.language_head = nn.Linear(hidden_dim, num_languages)
        self.switch_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        x = features.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.encoder(x)
        return {
            "language_logits": self.language_head(x),
            "switch_logits": self.switch_head(x).squeeze(-1),
        }


def lid_loss(
    outputs: dict[str, torch.Tensor],
    language_targets: torch.Tensor,
    switch_targets: torch.Tensor,
    switch_loss_weight: float = 0.3,
    switch_pos_weight: float = 1.0,
) -> torch.Tensor:
    language_logits = outputs["language_logits"]
    switch_logits = outputs["switch_logits"]

    language_loss = F.cross_entropy(
        language_logits.reshape(-1, language_logits.size(-1)),
        language_targets.reshape(-1),
        ignore_index=-100,
    )

    valid = switch_targets >= 0
    if valid.any():
        pos_weight = torch.tensor(float(max(switch_pos_weight, 1.0)), device=language_logits.device)
        switch_loss = F.binary_cross_entropy_with_logits(
            switch_logits[valid],
            switch_targets[valid].float(),
            pos_weight=pos_weight,
        )
    else:
        switch_loss = torch.zeros((), device=language_logits.device)

    return language_loss + switch_loss_weight * switch_loss
