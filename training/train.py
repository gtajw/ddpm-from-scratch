"""Train a DDPM from a YAML configuration."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from models import EMA, GaussianDiffusion, UNet
from utils.config import load_config, seed_everything
from utils.visualization import plot_losses, save_noising_steps, save_sample_grid

def build_dataloader(config: dict, train: bool = True) -> DataLoader:
    """Load MNIST or CIFAR-10, automatically downloading it when necessary."""
    channels, size = int(config["image_channels"]), int(config["image_size"])
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize((0.5,) * channels, (0.5,) * channels)])
    root = ROOT / config["data_dir"]
    dataset = datasets.MNIST(root, train=train, transform=transform, download=True) if config["dataset"].lower() == "mnist" else datasets.CIFAR10(root, train=train, transform=transform, download=True)
    return DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=train, num_workers=int(config["num_workers"]), pin_memory=torch.cuda.is_available(), persistent_workers=int(config["num_workers"]) > 0)

def build_model(config: dict) -> UNet:
    """Create a U-Net entirely from YAML hyperparameters."""
    return UNet(image_channels=int(config["image_channels"]), base_channels=int(config["base_channels"]), channel_multipliers=tuple(config["channel_multipliers"]), time_dim=int(config["time_dim"]), attention_resolution=int(config["attention_resolution"]), image_size=int(config["image_size"]), dropout=float(config["dropout"]))

def train(config: dict, resume: str | None = None) -> None:
    """Optimize epsilon prediction, maintain EMA, and save reproducible artifacts."""
    seed_everything(int(config["seed"])); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = ROOT / config["output_dir"]; output.mkdir(parents=True, exist_ok=True)
    loader = build_dataloader(config); model = build_model(config).to(device)
    diffusion = GaussianDiffusion(int(config["timesteps"]), config["beta_schedule"]).to(device)
    ema = EMA(model, float(config["ema_decay"])); optimizer = AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    losses: list[float] = []; start = 1
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False); model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"]); optimizer.load_state_dict(state["optimizer"]); losses = state.get("losses", []); start = state["epoch"] + 1
    print(f"Training on {device}.")
    for epoch in range(start, int(config["epochs"]) + 1):
        total = 0.0; model.train()
        for images, _ in tqdm(loader, desc=f"Epoch {epoch}/{config['epochs']}"):
            optimizer.zero_grad(set_to_none=True); loss = diffusion.training_loss(model, images.to(device, non_blocking=True)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])); optimizer.step(); ema.update(model); total += loss.item()
        losses.append(total / len(loader)); print(f"Epoch {epoch:03d} | loss: {losses[-1]:.6f}"); plot_losses(losses, output / "loss.png")
        if epoch % int(config["sample_every"]) == 0 or epoch == int(config["epochs"]):
            shape = (int(config["num_samples"]), int(config["image_channels"]), int(config["image_size"]), int(config["image_size"]))
            save_sample_grid(diffusion.sample(ema.ema_model.to(device), shape, device), output / "samples" / f"epoch_{epoch:03d}.png")
            save_noising_steps(diffusion, next(iter(loader))[0][:1].to(device), output / "noising_steps.png")
        if epoch % int(config["checkpoint_every"]) == 0 or epoch == int(config["epochs"]):
            path = output / "checkpoints" / f"epoch_{epoch:03d}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "losses": losses, "config": config}, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True); parser.add_argument("--resume")
    arguments = parser.parse_args(); train(load_config(arguments.config), arguments.resume)
