"""FGSM and four successors. All untargeted; inputs in [0, 1].

FGSM     Goodfellow et al., ICLR 2015      one gradient-sign step of size eps
BIM      Kurakin et al., ICLR-W 2017       FGSM repeated in small steps, clipped to the eps ball
PGD      Madry et al., ICLR 2018           BIM + random start inside the ball
MI-FGSM  Dong et al., CVPR 2018            BIM + momentum on the normalised gradient (transferability)
DeepFool Moosavi-Dezfooli et al., CVPR 2016  smallest L2 step to the nearest linearised boundary (no eps)
"""
import torch
import torch.nn.functional as F


def _grad(model, x, y):
    x = x.detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    return torch.autograd.grad(loss, x)[0]


def _project(x_adv, x, eps):
    return (x + (x_adv - x).clamp(-eps, eps)).clamp(0, 1).detach()


def fgsm(model, x, y, eps):
    return (x + eps * _grad(model, x, y).sign()).clamp(0, 1).detach()


def bim(model, x, y, eps, steps=10, alpha=None):
    alpha = alpha or eps / 4
    x_adv = x.clone()
    for _ in range(steps):
        x_adv = _project(x_adv + alpha * _grad(model, x_adv, y).sign(), x, eps)
    return x_adv


def pgd(model, x, y, eps, steps=10, alpha=None):
    alpha = alpha or eps / 4
    x_adv = _project(x + torch.empty_like(x).uniform_(-eps, eps), x, eps)
    for _ in range(steps):
        x_adv = _project(x_adv + alpha * _grad(model, x_adv, y).sign(), x, eps)
    return x_adv


def mifgsm(model, x, y, eps, steps=10, mu=1.0):
    alpha = eps / steps
    g = torch.zeros_like(x)
    x_adv = x.clone()
    for _ in range(steps):
        grad = _grad(model, x_adv, y)
        g = mu * g + grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        x_adv = _project(x_adv + alpha * g.sign(), x, eps)
    return x_adv


def deepfool(model, x, num_classes=10, overshoot=0.02, max_iter=50):
    """Returns x_adv. Moves each sample until its predicted class changes."""
    with torch.no_grad():
        k0 = model(x).argmax(1)
    r = torch.zeros_like(x)
    active = torch.ones(len(x), dtype=torch.bool, device=x.device)
    idx = torch.arange(len(x), device=x.device)
    for _ in range(max_iter):
        xi = (x + (1 + overshoot) * r).clamp(0, 1).detach().requires_grad_(True)
        logits = model(xi)
        active &= logits.argmax(1) == k0
        if not active.any():
            break
        grads = torch.stack([torch.autograd.grad(logits[:, c].sum(), xi, retain_graph=c < num_classes - 1)[0]
                             for c in range(num_classes)], 1)
        with torch.no_grad():
            f = logits - logits.gather(1, k0[:, None])
            w = grads - grads[idx, k0][:, None]
            wn = w.flatten(2).norm(dim=2).clamp_min(1e-12)
            dist = f.abs() / wn
            dist[idx, k0] = float("inf")
            c = dist.argmin(1)
            step = ((f[idx, c].abs() + 1e-4) / wn[idx, c] ** 2).view(-1, 1, 1, 1) * w[idx, c]
            r += step * active.view(-1, 1, 1, 1)
    return (x + (1 + overshoot) * r).clamp(0, 1).detach()


ATTACKS = {"FGSM": fgsm, "BIM": bim, "PGD": pgd, "MI-FGSM": mifgsm}
GRAD_CALLS = {"FGSM": 1, "BIM": 10, "PGD": 10, "MI-FGSM": 10}
