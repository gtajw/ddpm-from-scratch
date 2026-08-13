"""Visualization utilities for normalized DDPM images."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torchvision.utils import make_grid, save_image

def denormalize(images: Tensor) -> Tensor:
    """Map [-1, 1] tensors to displayable [0, 1]."""
    return images.detach().cpu().clamp(-1, 1).add(1).div(2)

def save_sample_grid(images: Tensor, path: str | Path, nrow: int = 8) -> None:
    """Save an image grid."""
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True); save_image(denormalize(images), destination, nrow=nrow)

def plot_losses(losses: Iterable[float], path: str | Path) -> None:
    """Save epoch loss history."""
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True); figure, axis = plt.subplots(figsize=(7, 4)); values = list(losses)
    axis.plot(range(1, len(values) + 1), values); axis.set(xlabel="Epoch", ylabel="MSE loss", title="DDPM training loss"); axis.grid(alpha=.25); figure.tight_layout(); figure.savefig(destination, dpi=150); plt.close(figure)

@torch.no_grad()
def save_noising_steps(diffusion: torch.nn.Module, image: Tensor, path: str | Path, steps: int = 8) -> None:
    """Save forward-noising states for a single image."""
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True); times = torch.linspace(0, diffusion.timesteps - 1, steps, device=image.device).long()
    save_image(make_grid(denormalize(torch.cat([diffusion.q_sample(image, time.unsqueeze(0)) for time in times])), nrow=steps), destination)
