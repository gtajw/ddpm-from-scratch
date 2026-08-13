# GitHub publishing checklist

Use this checklist after training and before making the repository public.

## Before the first commit

- [ ] `README.md` has no remaining `[placeholder]` values in the final Results table.
- [ ] `results/` contains the final sample grid, loss plot, and noising trajectory that the README embeds.
- [ ] Every reported FID includes its dataset, checkpoint, and number of generated images.
- [ ] You recorded hardware, training duration, seed, config filename, and Git commit in `results/README.md`.
- [ ] The notebook opens after changing its checkpoint path to a checkpoint you actually trained.
- [ ] You can run the MNIST command from a clean virtual environment.
- [ ] Python 3.14 can import both packages: `python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"`.
- [ ] On NVIDIA hardware, `torch.cuda.is_available()` is `True` before starting a long training run.
- [ ] The repository has an MIT `LICENSE` and `.gitignore` excludes data, outputs, environments, and checkpoints.

## Inspect what will be uploaded

```bash
git status
git diff --cached --check
git ls-files | grep -E '(^data/|^outputs/|\.pt$|\.pth$|\.env$)'
```

The final command should print nothing. On Windows without `grep`, inspect `git ls-files` manually or use:

```powershell
git ls-files | Select-String '^(data/|outputs/)|\.(pt|pth|env)$'
```

## Create the remote and push

1. On GitHub, create an empty repository named `ddpm-from-scratch`. Do not initialize it with a README or license because this repository already has both.
2. In the local repository, run:

   ```bash
   git init
   git add .
   git commit -m "Implement DDPM from scratch"
   git branch -M main
   git remote add origin https://github.com/<your-username>/ddpm-from-scratch.git
   git push -u origin main
   ```

3. On the repository page, verify rendered images, code blocks, links, and the license badge.
4. Add these topics: `ddpm`, `diffusion-models`, `pytorch`, `generative-ai`, `deep-learning`, `computer-vision`.
5. In the GitHub “About” field, use: “From-scratch PyTorch DDPM for MNIST and CIFAR-10, with EMA, FID, and reproducible training configs.”

## Final portfolio review

- [ ] The first screen of the README makes clear what you built and shows a real generated-image grid.
- [ ] Results are tied to a configuration and an evaluation protocol.
- [ ] The commit history contains source and small visual artifacts, not datasets or secrets.
- [ ] A recruiter can find installation, training, evaluation, and notebook instructions in under one minute.
- [ ] Your resume link points to the repository’s main page, not a local file or a single commit.
