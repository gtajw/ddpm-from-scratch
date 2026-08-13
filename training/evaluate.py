"""Generate samples and optionally calculate FID for a saved checkpoint."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import torch
from models import EMA, GaussianDiffusion
from training.train import build_dataloader, build_model
from utils.config import load_config
from utils.metrics import InceptionFeatureExtractor, frechet_distance
from utils.visualization import save_sample_grid

@torch.no_grad()
def main(config_path: str, checkpoint_path: str, num_images: int) -> None:
    """Load EMA weights, save samples, and compute FID against test images."""
    print("Loading config...")
    config = load_config(config_path); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading checkpoint...")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False); model = build_model(config).to(device); ema = EMA(model, float(config["ema_decay"])); ema.load_state_dict(state["ema"])
    print("Checkpoint loaded, building diffusion...")
    diffusion = GaussianDiffusion(int(config["timesteps"]), config["beta_schedule"]).to(device); output = ROOT / config["output_dir"] / "evaluation"; output.mkdir(parents=True, exist_ok=True)
    shape = (min(64, num_images), int(config["image_channels"]), int(config["image_size"]), int(config["image_size"]))
    print("Generating initial sample grid...")
    first_samples = diffusion.sample(ema.ema_model.to(device), shape, device); save_sample_grid(first_samples, output / "samples.png")
    print("Loading InceptionV3 model...")
    extractor = InceptionFeatureExtractor(device); real, generated, remaining = [], [], num_images
    loader = iter(build_dataloader(config, train=False))
    print(f"Generating and evaluating {num_images} images...")
    while remaining > 0:
        count = min(int(config["batch_size"]), remaining); images, _ = next(loader)
        samples = diffusion.sample(ema.ema_model, (count, shape[1], shape[2], shape[3]), device)
        real.append(extractor(images[:count]).cpu()); generated.append(extractor(samples).cpu()); remaining -= count
        print(f"Progress: {num_images - remaining}/{num_images} images processed")
    fid = frechet_distance(torch.cat(real).numpy(), torch.cat(generated).numpy()); (output / "fid.txt").write_text(f"FID ({num_images} images): {fid:.4f}\n", encoding="utf-8"); print((output / "fid.txt").read_text())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--num-images", type=int, default=1000)
    args = parser.parse_args(); main(args.config, args.checkpoint, args.num_images)
