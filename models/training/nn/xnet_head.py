"""PyTorch implementation of the XNet classification head with optional LoRA."""
from __future__ import annotations

import torch
import torch.nn as nn


class XNetHead(nn.Module):
    def __init__(self, hidden_size: int, n_tools: int, lora_r: int = 16, lora_alpha: int = 32):
        super().__init__()
        self.linear = nn.Linear(hidden_size, n_tools)
        # TODO: integrate LoRA

    def forward(self, hidden_states: torch.Tensor, **kwargs):  # noqa: D401
        return self.linear(hidden_states[:, 0, :])
