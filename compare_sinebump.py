"""
Compare the trained sine-bump wPINN(s) against the Godunov reference solution.

Regenerates:
  - comparison_sinebump.png           Godunov ref vs uniform wPINN
  - comparison_sinebump_adaptive.png  Godunov ref vs uniform vs adaptive
    (only if SineBump/adaptive/ModelSol.pkl exists)
and prints the per-slice relative-L1 error for each available variant.

    python3 compare_sinebump.py
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import godunov

TIMES = [0.0, 0.10, 0.25, 0.45]
# (label, path, matplotlib style)
VARIANTS = [
    ("uniform", "SineBump/best/ModelSol.pkl", "r--"),
    ("adaptive", "SineBump/adaptive/ModelSol.pkl", "g-."),
]


def load(path):
    m = torch.load(path, map_location="cpu", weights_only=False)
    m.eval()
    return m


def pred(model, x, t):
    inp = torch.tensor(np.column_stack([np.full_like(x, t), x]), dtype=torch.float32)
    with torch.no_grad():
        return model(inp).numpy().reshape(-1)


def slice_plot(x, ref, loaded, png):
    fig, axes = plt.subplots(1, len(TIMES), figsize=(5 * len(TIMES), 4.2), sharey=True)
    fig.suptitle("Sine-bump IC  u0=sin(pi(x+0.5)) on [-0.5,0.5] — Godunov reference vs wPINN", fontsize=13)
    for ax, t in zip(axes, TIMES):
        ax.plot(x, ref(x, t), "k-", lw=2.2, label="Godunov (ref)")
        for name, m, style in loaded:
            ax.plot(x, pred(m, x, t), style, lw=1.6, label=name)
        ax.set_title(f"t = {t}")
        ax.set_xlabel("x")
        ax.grid(True, ls=":")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("u")
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    print(f"saved {png}")


def main():
    available = [(n, load(p), s) for n, p, s in VARIANTS if os.path.exists(p)]
    if not available:
        raise SystemExit("No SineBump models found. Run train_sinebump.py first.")

    # Godunov reference field (ground truth — Burgers has no closed form here)
    xg, tv, U = godunov.godunov("SineBump", nx=2000, cfl=0.9)

    def ref(x, t):
        j = int(np.argmin(np.abs(tv - t)))
        return np.interp(x, xg, U[j])

    x = np.linspace(-1, 1, 1000)

    # uniform-only figure (kept for the baseline comparison)
    uni = [v for v in available if v[0] == "uniform"]
    if uni:
        slice_plot(x, ref, uni, "comparison_sinebump.png")
    # uniform-vs-adaptive overlay (only if both exist)
    if len(available) > 1:
        slice_plot(x, ref, available, "comparison_sinebump_adaptive.png")

    # per-slice relative L1 per variant
    print(f"\n{'variant':10s} {'relL1@.10':>10} {'relL1@.25':>10} {'relL1@.45':>10}")
    for name, m, style in available:
        row = [godunov.relative_l1_error(pred(m, x, t), ref(x, t)) for t in (0.10, 0.25, 0.45)]
        print(f"{name:10s} " + " ".join(f"{v:>10.4f}" for v in row))


if __name__ == "__main__":
    main()
