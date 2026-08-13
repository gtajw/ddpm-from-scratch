"""Time-conditioned U-Net used to predict diffusion noise."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    """Return a GroupNorm-compatible group count."""
    return max(group for group in range(1, min(32, channels) + 1) if channels % group == 0)


class SinusoidalTimeEmbedding(nn.Module):
    """Map integer diffusion steps to fixed sinusoidal vectors."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time embedding dimension must be an even value >= 4")
        self.dimension = dimension

    def forward(self, timesteps: Tensor) -> Tensor:
        """Return a sinusoidal embedding for a batch of timesteps."""
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / (half - 1)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class ResBlock(nn.Module):
    """A residual convolutional block modulated by a time embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        """Apply residual processing conditioned on ``time_embedding``."""
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Spatial self-attention with a residual connection."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Attend across all spatial positions."""
        batch, channels, height, width = x.shape
        h = self.norm(x).reshape(batch, channels, height * width).transpose(1, 2)
        h, _ = self.attention(h, h, h, need_weights=False)
        h = h.transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.projection(h)


class Downsample(nn.Module):
    """Halve spatial resolution using a strided convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Downsample the image feature map."""
        return self.conv(x)


class Upsample(nn.Module):
    """Double spatial resolution followed by a convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Upsample the image feature map."""
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """Compact DDPM U-Net with residual, time-conditioning, and attention blocks."""

    def __init__(
        self,
        image_channels: int,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 256,
        attention_resolution: int = 16,
        image_size: int = 32,
        dropout: float = 0.1,
    ) -> None:
        """Build the U-Net; image size must be divisible by 2^(levels - 1)."""
        super().__init__()
        if image_size % (2 ** (len(channel_multipliers) - 1)):
            raise ValueError("image_size is incompatible with the number of levels")
        self.input_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.down_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        skip_channels = [base_channels]
        current_channels = base_channels
        resolution = image_size
        self.down_attention = []
        for level, multiplier in enumerate(channel_multipliers):
            out_channels = base_channels * multiplier
            blocks = nn.ModuleList(
                [
                    ResBlock(current_channels, out_channels, time_dim, dropout),
                    ResBlock(out_channels, out_channels, time_dim, dropout),
                ]
            )
            self.down_blocks.append(blocks)
            skip_channels.extend([out_channels, out_channels])
            current_channels = out_channels
            self.down_attention.append(
                AttentionBlock(current_channels) if resolution <= attention_resolution else nn.Identity()
            )
            if level < len(channel_multipliers) - 1:
                self.downsample_layers.append(Downsample(current_channels))
                skip_channels.append(current_channels)
                resolution //= 2
        self.down_attention = nn.ModuleList(self.down_attention)
        self.middle_block1 = ResBlock(current_channels, current_channels, time_dim, dropout)
        self.middle_attention = AttentionBlock(current_channels)
        self.middle_block2 = ResBlock(current_channels, current_channels, time_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.up_attention = nn.ModuleList()
        for reverse_index, multiplier in enumerate(reversed(channel_multipliers)):
            level = len(channel_multipliers) - 1 - reverse_index
            out_channels = base_channels * multiplier
            blocks = nn.ModuleList()
            for _ in range(3):
                blocks.append(ResBlock(current_channels + skip_channels.pop(), out_channels, time_dim, dropout))
                current_channels = out_channels
            self.up_blocks.append(blocks)
            self.up_attention.append(
                AttentionBlock(current_channels) if resolution <= attention_resolution else nn.Identity()
            )
            if level > 0:
                self.upsample_layers.append(Upsample(current_channels))
                resolution *= 2
        if skip_channels:
            raise RuntimeError("U-Net skip channel construction is unbalanced")
        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(current_channels), current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, image_channels, 3, padding=1),
        )

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        """Predict the noise added to ``x`` at the supplied diffusion steps."""
        time_embedding = self.time_embedding(timesteps)
        h = self.input_conv(x)
        skips = [h]
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, time_embedding)
                skips.append(h)
            h = self.down_attention[level](h)
            if level < len(self.downsample_layers):
                h = self.downsample_layers[level](h)
                skips.append(h)
        h = self.middle_block1(h, time_embedding)
        h = self.middle_attention(h)
        h = self.middle_block2(h, time_embedding)
        for level, blocks in enumerate(self.up_blocks):
            for block in blocks:
                h = block(torch.cat((h, skips.pop()), dim=1), time_embedding)
            h = self.up_attention[level](h)
            if level < len(self.upsample_layers):
                h = self.upsample_layers[level](h)
        return self.output(h)
