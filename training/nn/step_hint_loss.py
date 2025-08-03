"""Auxiliary loss on tokens that follow <step_hint> according to the spec.
L_total = L_task + λ * L_hint + μ * L_GT
"""
from typing import Tuple

import torch
import torch.nn.functional as F


def step_hint_loss(task_logits: torch.Tensor, task_targets: torch.Tensor,
                   hint_logits: torch.Tensor, hint_targets: torch.Tensor,
                   gt_kl: torch.Tensor, lambda_hint: float = 0.3, mu_kl: float = 0.1) -> torch.Tensor:
    l_task = F.cross_entropy(task_logits, task_targets)
    l_hint = F.cross_entropy(hint_logits.flatten(0, 1), hint_targets.flatten())
    return l_task + lambda_hint * l_hint + mu_kl * gt_kl
