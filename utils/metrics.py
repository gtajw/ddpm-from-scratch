"""FID calculation using pretrained InceptionV3 features."""
from __future__ import annotations
import numpy as np
import torch
from scipy import linalg
from torch import Tensor, nn
from torchvision.models import Inception_V3_Weights, inception_v3

class InceptionFeatureExtractor(nn.Module):
    """ImageNet InceptionV3 penultimate feature extractor."""
    def __init__(self, device: torch.device) -> None:
        super().__init__(); self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1); self.model.fc = nn.Identity(); self.model.eval().to(device); self.device = device
    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """Extract 2048-dimensional features, expanding grayscale inputs."""
        images = images.to(self.device).clamp(-1, 1).add(1).div(2)
        if images.shape[1] == 1: images = images.repeat(1, 3, 1, 1)
        return self.model(torch.nn.functional.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False))

def frechet_distance(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    """Calculate Fréchet Inception Distance from two feature collections."""
    mean_a, cov_a = real_features.mean(0), np.cov(real_features, rowvar=False); mean_b, cov_b = generated_features.mean(0), np.cov(generated_features, rowvar=False); covariance = linalg.sqrtm(cov_a @ cov_b)
    if not np.isfinite(covariance).all(): covariance = linalg.sqrtm((cov_a + np.eye(cov_a.shape[0])*1e-6) @ (cov_b + np.eye(cov_b.shape[0])*1e-6))
    if np.iscomplexobj(covariance): covariance = covariance.real
    delta = mean_a - mean_b
    return float(delta @ delta + np.trace(cov_a + cov_b - 2 * covariance))
