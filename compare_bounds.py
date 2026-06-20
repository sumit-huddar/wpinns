"""
Compare the bounds variants on the moving shock against the exact solution:
  - old wPINN      (baseline, no bound enforcement) : ShockWave/best
  - bound penalty  (soft penalty in the loss)       : ShockWave/bound_penalty
  - hard bounds    (scaled-sigmoid output)          : ShockWave/hard_bounds

Regenerates comparison_shock_variants.png (3-way) and
comparison_shock_boundpenalty.png (baseline vs soft penalty), and prints the
undershoot/overshoot and relative-L1 errors.

    python3 compare_bounds.py
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from godunov import exact_solution, relative_l1_error

TIMES = [0.10, 0.25, 0.45]
CASE = "Moving"

VARIANTS = [
    ("old wPINN", "ShockWave/best/ModelSol.pkl", "r--"),
    ("bound penalty (soft)", "ShockWave/bound_penalty/ModelSol.pkl", "b-."),
    ("hard bounds", "ShockWave/hard_bounds/ModelSol.pkl", "g-"),
]


def load(path):
    m = torch.load(path, map_location="cpu", weights_only=False)
    m.eval()
    return m


def pred(model, x, t):
    inp = torch.tensor(np.column_stack([np.full_like(x, t), x]), dtype=torch.float32)
    with torch.no_grad():
        return model(inp).numpy().reshape(-1)


def slice_plot(x, loaded, title, png):
    fig, axes = plt.subplots(1, len(TIMES), figsize=(5 * len(TIMES), 4.2), sharey=True)
    fig.suptitle(title, fontsize=13)
    for ax, t in zip(axes, TIMES):
        ax.axhline(0, color="0.85", lw=1)
        ax.axhline(1, color="0.85", lw=1)
        ax.plot(x, exact_solution(CASE, x, t), "k-", lw=2.2, label="Exact")
        for name, m, style in loaded:
            ax.plot(x, pred(m, x, t), style, lw=1.6, label=name)
        ax.set_title(f"t = {t}")
        ax.set_xlabel("x")
        ax.grid(True, ls=":")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("u")
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    print(f"saved {png}")


def main():
    available = [(n, p, s) for n, p, s in VARIANTS if os.path.exists(p)]
    missing = [p for n, p, s in VARIANTS if not os.path.exists(p)]
    if missing:
        print(f"[warn] missing models: {missing}")
    loaded = [(n, load(p), s) for n, p, s in available]
    x = np.linspace(-1, 1, 1000)

    # 3-way comparison
    slice_plot(x, loaded, "Moving Shock — old wPINN vs soft bound penalty vs hard bounds",
               "comparison_shock_variants.png")
    # baseline vs soft penalty only (the earlier intermediate figure)
    two = [t for t in loaded if t[0] in ("old wPINN", "bound penalty (soft)")]
    if len(two) == 2:
        slice_plot(x, two, "Moving Shock — old wPINN vs wPINN + bound penalty",
                   "comparison_shock_boundpenalty.png")

    uex = exact_solution(CASE, x, 0.45)
    print(f"\n{'variant':22s} {'min u':>9} {'max u':>9} {'relL1@.45':>10}")
    for name, m, style in loaded:
        gmin, gmax = 1e9, -1e9
        for t in np.linspace(0.01, 0.45, 46):
            u = pred(m, x, t)
            gmin, gmax = min(gmin, u.min()), max(gmax, u.max())
        print(f"{name:22s} {gmin:+9.4f} {gmax:+9.4f} {relative_l1_error(pred(m, x, 0.45), uex):10.4f}")


if __name__ == "__main__":
    main()
