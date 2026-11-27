"""Show what each attack does to real CIFAR-10 images (eps = 8/255).

    python scripts/viz.py --data_root PATH

results/figures/adv_examples.png   adversarial images with the victim's prediction
results/figures/perturbations.png  the perturbation itself (x_adv - x), magnified
results/figures/how_it_works.png   original + perturbation = adversarial, for one image
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from src.attacks import ATTACKS, deepfool
from src.models import load_model

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(ROOT / "data"))
    ap.add_argument("--k", type=int, default=5, help="number of images (one per class)")
    ap.add_argument("--demo", type=int, default=1, help="which picked image to use in how_it_works.png")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    test = datasets.CIFAR10(args.data_root, train=False, download=True, transform=transforms.ToTensor())
    victim = load_model(ROOT / "checkpoints/resnet20_s0.pt", device)

    # first correctly classified image of each class, among the same 1,000 used in run.py
    picks, seen = [], set()
    for i in range(1000):
        img, lab = test[i]
        if lab in seen:
            continue
        with torch.no_grad():
            if victim(img[None].to(device)).argmax(1).item() == lab:
                picks.append(i); seen.add(lab)
        if len(picks) == args.k:
            break
    x = torch.stack([test[i][0] for i in picks]).to(device)
    y = torch.tensor([test[i][1] for i in picks], device=device)

    eps = 8 / 255
    advs = {"Original": x}
    for name, atk in ATTACKS.items():
        advs[name] = atk(victim, x, y, eps)
    advs["DeepFool"] = deepfool(victim, x)

    with torch.no_grad():
        probs = {n: victim(a).softmax(1) for n, a in advs.items()}

    names = list(advs)
    np_img = lambda t: t.permute(1, 2, 0).cpu().numpy()

    # figure 1: adversarial images + prediction
    fig, axes = plt.subplots(len(picks), len(names), figsize=(1.9 * len(names), 2.1 * len(picks)))
    for r in range(len(picks)):
        for c, n in enumerate(names):
            ax = axes[r, c]
            ax.imshow(np_img(advs[n][r]), interpolation="nearest")
            p = probs[n][r]; pred = p.argmax().item()
            ok = pred == y[r].item()
            ax.set_xlabel(f"{CLASSES[pred]} {p[pred]:.0%}", color="green" if ok else "red", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(n, fontsize=11)
    fig.suptitle("ResNet-20 prediction on adversarial images (eps = 8/255, DeepFool: no eps)\n"
                 "green = correct, red = fooled", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/adv_examples.png", dpi=150)

    # figure 2: perturbations, same scale for the eps-attacks, magnified to [-eps, eps] -> [0, 1]
    fig, axes = plt.subplots(len(picks), len(names) - 1, figsize=(1.9 * (len(names) - 1), 2.1 * len(picks)))
    for r in range(len(picks)):
        for c, n in enumerate(names[1:]):
            ax = axes[r, c]
            d = advs[n][r] - x[r]
            linf, l2 = d.abs().max().item() * 255, d.norm().item()
            ax.imshow(np_img((d / (2 * eps) + 0.5).clamp(0, 1)), interpolation="nearest")
            ax.set_xlabel(f"Linf {linf:.1f}/255  L2 {l2:.2f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(n, fontsize=11)
    fig.suptitle("Perturbation x_adv - x, magnified (gray = 0, full color = +-8/255)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/perturbations.png", dpi=150)
    # figure 3: original + perturbation = adversarial, one image, one row per attack
    r = args.demo
    label = lambda n: f"{CLASSES[probs[n][r].argmax().item()]} {probs[n][r].max():.0%}"
    fig = plt.figure(figsize=(7, 2.0 * (len(names) - 1)))
    gs = fig.add_gridspec(len(names) - 1, 5, width_ratios=[1, 0.25, 1, 0.25, 1])
    for i, n in enumerate(names[1:]):
        d = advs[n][r] - x[r]
        ok = probs[n][r].argmax().item() == y[r].item()
        panels = [
            (0, x[r], "original x", label("Original"), "green"),
            (2, (d / (2 * eps) + 0.5).clamp(0, 1), "perturbation (magnified)",
             f"Linf {d.abs().max().item() * 255:.1f}/255  L2 {d.norm().item():.2f}", "black"),
            (4, advs[n][r], "adversarial x_adv", label(n), "green" if ok else "red"),
        ]
        for c, img, title, xl, color in panels:
            ax = fig.add_subplot(gs[i, c])
            ax.imshow(np_img(img), interpolation="nearest")
            ax.set_xlabel(xl, color=color, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(n, fontsize=11)
            if i == 0:
                ax.set_title(title, fontsize=10)
        for c, sym in ((1, "+"), (3, "=")):
            ax = fig.add_subplot(gs[i, c]); ax.axis("off")
            ax.text(0.5, 0.5, sym, ha="center", va="center", fontsize=22)
    fig.suptitle("x + perturbation = x_adv  (eps = 8/255, DeepFool: no eps)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/how_it_works.png", dpi=150)
    print("picked test indices:", picks)


if __name__ == "__main__":
    main()
