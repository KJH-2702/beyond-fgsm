"""Train the ResNet-20 used as the attack target (plain cross-entropy baseline).

    python scripts/train.py --seed 0 --out checkpoints/my_s0.pt --data_root PATH

Same recipe as the checkpoints in checkpoints/ (trained on an NVIDIA A100):
CIFAR-10 train set (50,000), 100 epochs, batch 128, SGD (momentum 0.9, Nesterov),
lr 0.05 with per-step cosine decay, weight decay 5e-4, RandomCrop(32, pad 4) +
horizontal flip, inputs in [0, 1] (no normalisation).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models import ResNet20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", default=None, help="default: checkpoints/trained_s{seed}.pt")
    args = ap.parse_args()
    out = args.out or ROOT / f"checkpoints/trained_s{args.seed}.pt"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    train = datasets.CIFAR10(args.data_root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(args.data_root, train=False, download=True, transform=transforms.ToTensor())
    train_dl = DataLoader(train, batch_size=128, shuffle=True, num_workers=4, drop_last=False)
    test_dl = DataLoader(test, batch_size=500)

    model = ResNet20().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(x)
        print(f"epoch {epoch}/{args.epochs}  loss={total / len(train):.4f}")

    model.eval()
    with torch.no_grad():
        correct = sum((model(x.to(device)).argmax(1) == y.to(device)).sum().item() for x, y in test_dl)
    acc = correct / len(test)
    print(f"test acc = {acc:.4f}")
    torch.save({"state_dict": model.state_dict(), "metrics": {"clean_acc": acc}}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
