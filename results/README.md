# Results record

Use this file as the source of truth for every figure and number that appears in the root README. Fill it in while training, not from memory afterward.

## Run metadata

```text
Dataset: CIFAR-10
Config file and Git commit: configs/cifar10.yaml
Checkpoint evaluated: epoch_030.pt
Date run: July 22, 2026
Operating system: Windows
GPU / accelerator: NVIDIA GeForce GTX 1650 (Max-Q Design)
CUDA and PyTorch versions: CUDA available, PyTorch 2.3+
Batch size: 128
Epochs completed: 30
Diffusion timesteps and beta schedule: 650 timesteps, cosine schedule
EMA decay: 0.9999
Random seed: 42
Training duration: ~3.5 hours
Evaluation image count: 5000
FID from outputs/cifar10/evaluation/fid.txt: 230.4882
```

## Files to commit here

- `mnist_final_samples.png` or `cifar10_final_samples.png`: a final 8×8 EMA grid.
- `<dataset>_loss.png`: the full loss curve from the same run.
- `<dataset>_noising_steps.png`: one forward-diffusion visualization.
- Optionally, one compact `comparison.png` with an early and late sample grid side-by-side.

Do not commit raw datasets, every intermediate grid, TensorBoard logs, or `.pt` checkpoints to the normal repository history. If you decide a trained checkpoint is valuable to share, attach it to a GitHub Release and state its filename, SHA-256 checksum, config, and epoch here.

## Result-writing rules

1. State the FID sample count next to every FID value.
2. State whether samples came from EMA weights.
3. Do not cherry-pick a grid without identifying the checkpoint/epoch.
4. Describe limitations plainly; recognizable CIFAR-10 structure is more meaningful than an unqualified “high quality” claim.
5. Keep the root README’s Results table and this record consistent.
