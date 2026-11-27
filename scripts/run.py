"""Run all experiments and write results/tables/*.csv and results/figures/attack_curve.png.

    python scripts/run.py                      # downloads CIFAR-10 into ./data if missing
    python scripts/run.py --data_root PATH     # reuse an existing CIFAR-10 folder

Victim: ResNet-20 trained on full CIFAR-10 (seed 0). Seeds 1 and 2 are the
transfer targets. First N test images, fixed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from src.attacks import ATTACKS, GRAD_CALLS, deepfool
from src.models import load_model

EPS_255 = [0, 1, 2, 4, 8, 16]


def accuracy(model, x, y, bs=500):
    with torch.no_grad():
        return sum((model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
                   for i in range(0, len(x), bs)) / len(x)


def batched(fn, x, y, bs=250):
    return torch.cat([fn(x[i:i + bs], y[i:i + bs]) for i in range(0, len(x), bs)])


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(ROOT / "data"))
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    test = datasets.CIFAR10(args.data_root, train=False, download=True, transform=transforms.ToTensor())
    x = torch.stack([test[i][0] for i in range(args.n)]).to(device)
    y = torch.tensor([test[i][1] for i in range(args.n)], device=device)

    victim = load_model(ROOT / "checkpoints/resnet20_s0.pt", device)
    others = {s: load_model(ROOT / f"checkpoints/resnet20_{s}.pt", device) for s in ("s1", "s2")}
    print(f"device={device}  n={args.n}  clean acc={accuracy(victim, x, y):.3f}")

    # 1) attack-strength curve: accuracy under attack vs eps (white-box, victim)
    curve = []
    for name, atk in ATTACKS.items():
        for e in EPS_255:
            eps = e / 255
            x_adv = x if e == 0 else batched(lambda a, b: atk(victim, a, b, eps), x, y)
            acc = accuracy(victim, x_adv, y)
            curve.append({"attack": name, "eps_255": e, "acc": round(acc, 4)})
            print(f"[curve] {name:8s} eps={e:2d}/255  acc={acc:.3f}")

    # 2) cost and 3) transfer, at eps = 8/255
    eps = 8 / 255
    summary = []
    for name, atk in ATTACKS.items():
        sync(device); t0 = time.perf_counter()
        x_adv = batched(lambda a, b: atk(victim, a, b, eps), x, y)
        sync(device); sec = time.perf_counter() - t0
        row = {"attack": name, "grad_calls": GRAD_CALLS[name], "sec_per_1000": round(sec * 1000 / args.n, 2),
               "white_box_acc": round(accuracy(victim, x_adv, y), 4)}
        for s, m in others.items():
            row[f"transfer_acc_{s}"] = round(accuracy(m, x_adv, y), 4)
        row["mean_L2"] = round((x_adv - x).flatten(1).norm(dim=1).mean().item(), 4)
        summary.append(row)
        print("[eps 8/255]", row)

    # 4) DeepFool: no eps; how small can a successful perturbation be?
    sync(device); t0 = time.perf_counter()
    x_df = batched(lambda a, b: deepfool(victim, a), x, y, bs=100)
    sync(device); sec = time.perf_counter() - t0
    with torch.no_grad():
        clean_pred = torch.cat([victim(x[i:i + 500]).argmax(1) for i in range(0, len(x), 500)])
        df_pred = torch.cat([victim(x_df[i:i + 500]).argmax(1) for i in range(0, len(x), 500)])
    ok = (clean_pred == y) & (df_pred != clean_pred)
    d = (x_df - x)[ok]
    df = {"attack": "DeepFool", "n_correct_clean": int((clean_pred == y).sum()), "n_fooled": int(ok.sum()),
          "median_L2": round(d.flatten(1).norm(dim=1).median().item(), 4),
          "median_Linf_255": round(d.flatten(1).abs().amax(dim=1).median().item() * 255, 2),
          "sec_per_1000": round(sec * 1000 / args.n, 2)}
    print("[deepfool]", df)

    # write
    for fname, rows in (("curve.csv", curve), ("eps8_summary.csv", summary), ("deepfool.csv", [df])):
        with open(ROOT / "results/tables" / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    plt.figure(figsize=(5.5, 4))
    for name in ATTACKS:
        pts = [r for r in curve if r["attack"] == name]
        plt.plot([r["eps_255"] for r in pts], [r["acc"] for r in pts], marker="o", label=name)
    plt.xlabel("epsilon (x/255, L-inf)"); plt.ylabel("accuracy under attack")
    plt.title("ResNet-20 / CIFAR-10, white-box"); plt.xticks(EPS_255); plt.ylim(0, 1)
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(ROOT / "results/figures/attack_curve.png", dpi=150)


if __name__ == "__main__":
    main()
