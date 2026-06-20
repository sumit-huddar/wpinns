# Adaptive collocation sampling — wPINNs on Burgers

Does concentrating collocation points where the model's PDE residual is largest
help wPINNs? Tested on two Riemann problems on `x ∈ [−1,1]`, `t ∈ [0, 0.45]`:
the **moving shock** (`u₀ = 1, 0`) and the **rarefaction** (`u₀ = −1, +1`).

This experiment is on the **original solver** (forked from `main`) — no bounds or
other changes — so the only difference between the two runs in each A/B is the
collocation sampling.

## Method

Every `RESAMPLE_FREQ` (250) epochs, the collocation set is rebuilt:

1. Draw a large uniform candidate pool (10× `N_coll`).
2. Score each point by the **pointwise strong-form residual** `|u_t + u·u_x|`
   (`pointwise_residual`). This peaks where the solution is hardest to fit — the
   shock, or the rarefaction's fan-edge kinks.
3. Resample: `RESAMPLE_UNIFORM_FRAC` (0.5) kept uniform for coverage, the rest
   sampled **∝ residual** (`resample_collocation`).
4. `fit()` re-reads the loader each epoch, so the points track the feature as it
   moves and sharpens.

Opt-in via `resample_freq` (0 = original uniform behaviour). Toggle/case set in
`train_shock.py` (`USE_ADAPTIVE_SAMPLING`, `CASE`) or via `WPINN_ADAPTIVE` /
`WPINN_CASE` env vars.

## Results (3000 epochs, identical config per case, only sampling differs)

| Case | uniform rel L1 | adaptive rel L1 | effect |
|---|---|---|---|
| **Moving shock** | 0.0354 | **0.0056** | **~6× better** ✅ |
| **Rarefaction**  | **0.0191** | 0.0311 | ~1.6× worse ❌ |

Shock, global over the space–time domain:

| Variant | undershoot (min u) | overshoot (max u) | rel L1 |
|---|---|---|---|
| uniform  | −0.233 | +1.030 | 0.0354 |
| adaptive | **−0.009** | **+1.009** | **0.0056** |

Plots: `comparison_shock_adaptive.png`, `comparison_rarefaction_adaptive.png`.

## Conclusion

**Residual-adaptive sampling is feature-dependent — it helps exactly when there
is a localized sharp feature to resolve, and hurts on smooth solutions.**

- **Shock:** a discontinuity is a single sharp spike of residual. Concentrating
  points there resolves the jump, fixes its position, and — notably — nearly
  removes the over/undershoot **with no bound constraint** (min −0.233 → −0.009).
  A different, more fundamental route to the physical-bounds goal: resolve the
  shock properly and the spurious oscillation largely disappears on its own.
- **Rarefaction:** the solution is smooth (a linear fan); the residual is weak
  and spread out. Pulling points to the mild fan-edge kinks starves the smooth
  interior, which depends on uniform coverage, adding wiggles and raising error.

So adaptive sampling is a tool for discontinuities, not a free win to apply
blindly.

## Reproduce

```bash
# uniform vs adaptive, per case (writes ShockWave/{uniform,adaptive} or
# RarefactionWave/{uniform,adaptive}):
WPINN_CASE=Moving      WPINN_ADAPTIVE=0 python3 train_shock.py
WPINN_CASE=Moving      WPINN_ADAPTIVE=1 python3 train_shock.py
WPINN_CASE=Rarefaction WPINN_ADAPTIVE=0 python3 train_shock.py
WPINN_CASE=Rarefaction WPINN_ADAPTIVE=1 python3 train_shock.py
```
