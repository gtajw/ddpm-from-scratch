"""Gaussian forward and reverse processes for DDPM."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _extract(values: Tensor, timesteps: Tensor, target_shape: torch.Size) -> Tensor:
    """Gather per-timestep scalars and reshape them to broadcast over images."""
    gathered = values.gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))


def linear_beta_schedule(timesteps: int) -> Tensor:
    """Return the original DDPM linear beta schedule, scaled to any length."""
    scale = 1000 / timesteps
    return torch.linspace(scale * 1e-4, scale * 0.02, timesteps, dtype=torch.float32).clamp(max=0.999)


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> Tensor:
    """Return the cosine schedule proposed by Nichol and Dhariwal."""
    steps = timesteps + 1
    positions = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    cumulative = torch.cos(((positions / timesteps) + offset) / (1 + offset) * math.pi / 2).pow(2)
    cumulative = cumulative / cumulative[0]
    return (1 - (cumulative[1:] / cumulative[:-1])).clamp(1e-5, 0.999)


class GaussianDiffusion(nn.Module):
    """Precompute DDPM coefficients and perform training or ancestral sampling."""

    def __init__(self, timesteps: int = 1000, schedule: str = "linear") -> None:
        """Create a diffusion process with a linear or cosine variance schedule."""
        super().__init__()
        if schedule not in {"linear", "cosine"}:
            raise ValueError("schedule must be 'linear' or 'cosine'")
        betas = linear_beta_schedule(timesteps) if schedule == "linear" else cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_previous = torch.cat((torch.ones(1), alpha_cumprod[:-1]))
        posterior_variance = betas * (1 - alpha_cumprod_previous) / (1 - alpha_cumprod)
        for name, value in {
            "betas": betas,
            "alphas": alphas,
            "alpha_cumprod": alpha_cumprod,
            "alpha_cumprod_previous": alpha_cumprod_previous,
            "sqrt_alpha_cumprod": torch.sqrt(alpha_cumprod),
            "sqrt_one_minus_alpha_cumprod": torch.sqrt(1 - alpha_cumprod),
            "sqrt_recip_alphas": torch.sqrt(1 / alphas),
            "posterior_variance": posterior_variance,
        }.items():
            self.register_buffer(name, value)
        self.timesteps = timesteps

    def q_sample(self, x_start: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        """Draw q(x_t | x_0) using its closed-form reparameterization."""
        noise = torch.randn_like(x_start) if noise is None else noise
        return (
            _extract(self.sqrt_alpha_cumprod, timesteps, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_start.shape) * noise
        )

    def training_loss(self, model: nn.Module, x_start: Tensor) -> Tensor:
        """Compute the epsilon-prediction objective for a batch of images."""
        timesteps = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device)
        noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, timesteps, noise)
        return torch.nn.functional.mse_loss(model(noisy, timesteps), noise)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: Tensor, timesteps: Tensor) -> Tensor:
        """Sample one reverse transition p(x_{t-1} | x_t)."""
        beta = _extract(self.betas, timesteps, x.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x.shape)
        sqrt_recip_alpha = _extract(self.sqrt_recip_alphas, timesteps, x.shape)
        predicted_noise = model(x, timesteps)
        mean = sqrt_recip_alpha * (x - beta * predicted_noise / sqrt_one_minus)
        variance = _extract(self.posterior_variance, timesteps, x.shape)
        noise = torch.randn_like(x)
        nonzero = (timesteps != 0).float().reshape(x.shape[0], *((1,) * (x.ndim - 1)))
        return mean + nonzero * torch.sqrt(variance.clamp_min(1e-20)) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int, int, int, int], device: torch.device) -> Tensor:
        """Generate images by iterating the learned reverse Markov chain."""
        x = torch.randn(shape, device=device)
        was_training = model.training
        model.eval()
        for i, step in enumerate(reversed(range(self.timesteps))):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
            if (i + 1) % 100 == 0 or i == self.timesteps - 1:
                print(f"  Sampling progress: {i + 1}/{self.timesteps} timesteps")
        model.train(was_training)
        return x.clamp(-1, 1)
