# Sine-bump initial condition — wPINNs on Burgers

How well does the wPINN handle a richer, non-textbook initial condition, and does
residual-adaptive collocation sampling help?

## Initial condition

A half-sine hump in the middle of the domain, zero elsewhere:

```
u0(x) = sin(pi*(x + 0.5))   for |x| <= 0.5   (peak 1 at x=0, 0 at x=+-0.5)
u0(x) = 0                   otherwise
```

on `x in [-1, 1]`, `t in [0, 0.45]`. As a Burgers wave the hump **steepens into a
shock on its trailing (right) edge** while the leading (left) edge **rarefies**.
There is no closed-form entropy solution, so a fine-grid **Godunov** run
(`godunov.py`, case `SineBump`) is the ground-truth reference.

Models / scripts:
- `train_sinebump.py` — trains the wPINN (3000 epochs); `WPINN_ADAPTIVE=1` for the
  adaptive variant. Writes `SineBump/best` (uniform) or `SineBump/adaptive`.
- `compare_sinebump.py` — regenerates the plots and per-slice errors.

## Result 1 — wPINN on the sine-bump IC (uniform sampling)

Global rel L1 = **0.090** vs the Godunov reference. The error is **dominated by the
forming shock** and grows with time as the shock sharpens:

| t | rel L1 |
|---|---|
| 0.10 | 0.042 |
| 0.25 | 0.101 |
| 0.45 | 0.164 |

The wPINN captures the hump and the rarefying left side well, but **smears the shock
and lags its position** (peak at x≈0.35 vs the true x≈0.55 at t=0.45). It stays
physical (`u` in ~[0, 1]). See `comparison_sinebump.png`.

## Result 2 — residual-adaptive sampling makes it WORSE

Applying the residual-adaptive sampler (which gave a ~6× win on the pure moving
shock) **hurts** here:

| variant | rel L1 @ .10 | rel L1 @ .25 | rel L1 @ .45 | global rel L1 |
|---|---|---|---|---|
| uniform  | 0.042 | 0.101 | 0.164 | **0.090** |
| adaptive | 0.209 | 0.327 | 0.260 | 0.255 (~3x worse) |

The adaptive solution is **wildly oscillatory** across the whole domain (see
`comparison_sinebump_adaptive.png`). Concentrating points on the forming shock
**starves the bump's broad smooth structure** — the hump, the rarefying left side,
the flat zeros — which then fills with Gibbs-like oscillations. Those cost far more
than sharpening the shock saves.

## Conclusion — when adaptive sampling helps

Across three solution types tested in this project:

| Solution | Character | Adaptive effect |
|---|---|---|
| Moving shock | near-trivial field + 1 shock | ~6x better |
| Rarefaction | all smooth | ~1.6x worse |
| **Sine bump** | rich smooth structure + forming shock | **~3x worse** |

The deciding factor is **not** simply "is there a discontinuity," but the **ratio of
sharp localized feature to smooth structure that needs coverage**:

- **Near-trivial field with one sharp feature** → adaptive sampling wins big (all the
  spare resolution goes to the only hard region).
- **Substantial smooth structure** → concentrating points starves it and backfires,
  whether or not a shock is also present.

So adaptive collocation sampling is a targeted tool for shock-dominated problems on
otherwise-trivial fields, not a general-purpose improvement.

## Reproduce

```bash
python3 train_sinebump.py                    # uniform  -> SineBump/best
WPINN_ADAPTIVE=1 python3 train_sinebump.py   # adaptive -> SineBump/adaptive
python3 compare_sinebump.py                  # plots + error tables
```
