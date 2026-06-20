"""
Compare uniform vs residual-adaptive collocation sampling for both the moving
shock and the rarefaction, against the exact solution.

Regenerates comparison_shock_adaptive.png / comparison_rarefaction_adaptive.png
and prints the undershoot/overshoot and relative-L1 errors. Run after the four
training runs have produced their models:

    python3 compare_adaptive.py

Models expected at:
    ShockWave/{uniform,adaptive}/ModelSol.pkl
    RarefactionWave/{uniform,adaptive}/ModelSol.pkl
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from godunov import exact_solution, relative_l1_error

TIMES = [0.10, 0.25, 0.45]

# (case_label, godunov case name, base dir, output png)
CASES = [
    ("Moving Shock", "Moving", "ShockWave", "comparison_shock_adaptive.png"),
    ("Rarefaction", "Rarefaction", "RarefactionWave", "comparison_rarefaction_adaptive.png"),
]


def load(path):
    m = torch.load(path, map_location="cpu", weights_only=False)
    m.eval()
    return m


def pred(model, x, t):
    inp = torch.tensor(np.column_stack([np.full_like(x, t), x]), dtype=torch.float32)
    with torch.no_grad():
        return model(inp).numpy().reshape(-1)


def main():
    x = np.linspace(-1, 1, 1000)
    for label, case, base, png in CASES:
        paths = {"uniform": f"{base}/uniform/ModelSol.pkl",
                 "adaptive": f"{base}/adaptive/ModelSol.pkl"}
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"[skip] {label}: missing {[p for p in paths.values() if not os.path.exists(p)]}")
            continue
        models = {k: load(p) for k, p in paths.items()}
        styles = {"uniform": "r--", "adaptive": "g-"}

        fig, axes = plt.subplots(1, len(TIMES), figsize=(5 * len(TIMES), 4.2), sharey=True)
        fig.suptitle(f"{label} — uniform vs residual-adaptive collocation sampling", fontsize=13)
        for ax, t in zip(axes, TIMES):
            ax.plot(x, exact_solution(case, x, t), "k-", lw=2.2, label="Exact")
            for k, m in models.items():
                ax.plot(x, pred(m, x, t), styles[k], lw=1.6, label=k)
            ax.set_title(f"t = {t}")
            ax.set_xlabel("x")
            ax.grid(True, ls=":")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("u")
        plt.tight_layout()
        plt.savefig(png, dpi=150)
        print(f"\nsaved {png}")

        uex = exact_solution(case, x, 0.45)
        print(f"{label}")
        print(f"  {'variant':10s} {'min u':>9} {'max u':>9} {'relL1@.45':>10}")
        for k, m in models.items():
            gmin, gmax = 1e9, -1e9
            for t in np.linspace(0.01, 0.45, 46):
                u = pred(m, x, t)
                gmin, gmax = min(gmin, u.min()), max(gmax, u.max())
            print(f"  {k:10s} {gmin:+9.4f} {gmax:+9.4f} {relative_l1_error(pred(m, x, 0.45), uex):10.4f}")


if __name__ == "__main__":
    main()
