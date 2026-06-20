"""
Baseline vs optimal wPINN on the moving shock — the analysis behind
optimal_wpinn.ipynb, as a standalone script so the computation is visible and
reproducible.

  - global relative L1 (over the space-time domain) for each model
  - the headline improvement (% and factor)
  - per-time-slice relative L1
  - physical-bounds check (min/max u vs [0, 1])
  - figures: optimal_wpinn_slices.png, optimal_wpinn_error_vs_time.png

Run after train_optimal.py has produced both models:

    MODE=baseline python3 train_optimal.py
    MODE=optimal  python3 train_optimal.py
    python3 compare_optimal.py
"""

import os
import numpy as np
import torch

from godunov import exact_solution, relative_l1_error

# matplotlib is imported inside main() so that importing these helpers from a
# notebook does not force the non-interactive Agg backend.

CASE = "Moving"
MODELS = {
    "baseline": "ShockWave/baseline/ModelSol.pkl",
    "optimal":  "ShockWave/optimal/ModelSol.pkl",
}
SLICE_TIMES = [0.10, 0.25, 0.45]


def load(path):
    m = torch.load(path, map_location="cpu", weights_only=False)
    m.eval()
    return m


def pred(model, x, t):
    inp = torch.tensor(np.column_stack([np.full_like(x, t), x]), dtype=torch.float32)
    with torch.no_grad():
        return model(inp).numpy().reshape(-1)


def global_rel_l1(model, x, ts):
    """Relative L1 over the whole (t, x) grid."""
    num = den = 0.0
    for t in ts:
        ue = exact_solution(CASE, x, t)
        up = pred(model, x, t)
        num += np.sum(np.abs(up - ue))
        den += np.sum(np.abs(ue))
    return num / den


def main():
    missing = [p for p in MODELS.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Missing models: {missing}. Run train_optimal.py first.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = {k: load(p) for k, p in MODELS.items()}
    x = np.linspace(-1, 1, 1000)
    ts = np.linspace(0.01, 0.45, 45)

    # ── headline improvement ────────────────────────────────────────────────
    err = {k: global_rel_l1(m, x, ts) for k, m in models.items()}
    improvement = 100.0 * (err["baseline"] - err["optimal"]) / err["baseline"]
    factor = err["baseline"] / err["optimal"]
    print("=== global relative L1 (whole domain) ===")
    print(f"  baseline = {err['baseline']:.4f}")
    print(f"  optimal  = {err['optimal']:.4f}")
    print(f"  improvement = {improvement:.1f}%   ({factor:.1f}x lower error)")

    # ── per-slice relative L1 ───────────────────────────────────────────────
    print("\n=== relative L1 per time slice ===")
    print(f"  {'t':>6} {'baseline':>10} {'optimal':>10}")
    for t in SLICE_TIMES:
        eb = relative_l1_error(pred(models["baseline"], x, t), exact_solution(CASE, x, t))
        eo = relative_l1_error(pred(models["optimal"], x, t), exact_solution(CASE, x, t))
        print(f"  {t:>6} {eb:>10.4f} {eo:>10.4f}")

    # ── physical-bounds check ───────────────────────────────────────────────
    print("\n=== physical bounds (band [0, 1]) ===")
    for k, m in models.items():
        gmin = min(pred(m, x, t).min() for t in ts)
        gmax = max(pred(m, x, t).max() for t in ts)
        print(f"  {k:10s} u range [{gmin:+.4f}, {gmax:+.4f}]")

    # ── figure 1: solution slices ───────────────────────────────────────────
    fig, axes = plt.subplots(1, len(SLICE_TIMES), figsize=(5 * len(SLICE_TIMES), 4.2), sharey=True)
    fig.suptitle("Moving shock — baseline vs optimal (hard bounds + adaptive sampling)", fontsize=13)
    for ax, t in zip(axes, SLICE_TIMES):
        ax.axhline(0, color="0.85", lw=1)
        ax.axhline(1, color="0.85", lw=1)
        ax.plot(x, exact_solution(CASE, x, t), "k-", lw=2.2, label="Exact")
        ax.plot(x, pred(models["baseline"], x, t), "r--", lw=1.5, label="baseline")
        ax.plot(x, pred(models["optimal"], x, t), "g-", lw=1.6, label="optimal")
        ax.set_title(f"t = {t}")
        ax.set_xlabel("x")
        ax.grid(True, ls=":")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("u")
    plt.tight_layout()
    plt.savefig("optimal_wpinn_slices.png", dpi=150)
    print("\nsaved optimal_wpinn_slices.png")

    # ── figure 2: error vs time ─────────────────────────────────────────────
    tt = np.linspace(0.02, 0.45, 40)
    plt.figure(figsize=(7, 4))
    for k, style in [("baseline", "r--"), ("optimal", "g-")]:
        e = [relative_l1_error(pred(models[k], x, t), exact_solution(CASE, x, t)) for t in tt]
        plt.plot(tt, e, style, label=k)
    plt.xlabel("t"); plt.ylabel("relative L1")
    plt.title("Moving shock — relative L1 error vs time")
    plt.legend(); plt.grid(True, ls=":")
    plt.tight_layout()
    plt.savefig("optimal_wpinn_error_vs_time.png", dpi=150)
    print("saved optimal_wpinn_error_vs_time.png")


if __name__ == "__main__":
    main()
