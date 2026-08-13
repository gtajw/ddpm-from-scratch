"""Exponential moving average utilities for diffusion models."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class EMA:
    """Maintain a non-trainable exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        """Create an EMA copy of ``model``."""
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be between zero and one")
        self.decay = decay
        self.ema_model = deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the moving average after one optimizer step."""
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            source = model_state[key].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)

    def state_dict(self) -> dict:
        """Return serializable EMA state."""
        return {"decay": self.decay, "model": self.ema_model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore a serialized EMA state."""
        self.decay = float(state["decay"])
        self.ema_model.load_state_dict(state["model"])
