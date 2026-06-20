"""
Compare the trained sine-bump wPINN against the Godunov reference solution.

Regenerates comparison_sinebump.png and prints the L1 / relative-L1 errors.
Run after train_sinebump.py has produced SineBump/best/ModelSol.pkl:

    python3 compare_sinebump.py
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import godunov

MODEL = "SineBump/best/ModelSol.pkl"
TIMES = [0.0, 0.10, 0.25, 0.45]


def main():
    if not os.path.exists(MODEL):
        raise SystemExit(f"Model not found at '{MODEL}'. Run train_sinebump.py first.")

    # Godunov reference field (ground truth — Burgers has no closed form here)
    xg, tv, U = godunov.godunov("SineBump", nx=2000, cfl=0.9)

    def ref(x, t):
        j = int(np.argmin(np.abs(tv - t)))
        return np.interp(x, xg, U[j])

    model = torch.load(MODEL, map_location="cpu", weights_only=False)
    model.eval()

    def pred(x, t):
        inp = torch.tensor(np.column_stack([np.full_like(x, t), x]), dtype=torch.float32)
        with torch.no_grad():
            return model(inp).numpy().reshape(-1)

    x = np.linspace(-1, 1, 1000)

    fig, axes = plt.subplots(1, len(TIMES), figsize=(5 * len(TIMES), 4.2), sharey=True)
    fig.suptitle("Sine-bump IC  u0=sin(pi(x+0.5)) on [-0.5,0.5] — Godunov reference vs wPINN", fontsize=13)
    for ax, t in zip(axes, TIMES):
        ax.plot(x, ref(x, t), "k-", lw=2.2, label="Godunov (ref)")
        ax.plot(x, pred(x, t), "b--", lw=1.6, label="wPINN")
        ax.set_title(f"t = {t}")
        ax.set_xlabel("x")
        ax.grid(True, ls=":")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("u")
    plt.tight_layout()
    plt.savefig("comparison_sinebump.png", dpi=150)
    print("saved comparison_sinebump.png")

    # global range + per-slice relative L1
    gmin, gmax = 1e9, -1e9
    for t in np.linspace(0.01, 0.45, 46):
        u = pred(x, t)
        gmin, gmax = min(gmin, u.min()), max(gmax, u.max())
    print(f"wPINN global u range: [{gmin:.3f}, {gmax:.3f}]  (physical band is [0, 1])")
    print(f"{'t':>6} {'rel L1':>10}")
    for t in [0.10, 0.25, 0.45]:
        r, u = ref(x, t), pred(x, t)
        rel = godunov.relative_l1_error(u, r)
        print(f"{t:>6} {rel:>10.4f}")


if __name__ == "__main__":
    main()
