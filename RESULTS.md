# Boundedness experiment — wPINNs on the moving shock

Burgers' moving shock: `u₀ = 1 (x≤0), 0 (x>0)`, domain `x ∈ [−1,1]`, `t ∈ [0, 0.45]`.
The exact entropy solution is a sharp step that moves right at speed ½ and stays
within the physical band `u ∈ [0, 1]` (maximum principle).

## Motivation

The baseline wPINN reproduces the shock but **undershoots below 0** near the jump
(down to ≈ −0.37), which is unphysical — classical solvers (Lax–Friedrichs,
Godunov) never leave `[0, 1]`. This experiment tests two ways of enforcing the
bound and compares them against the baseline.

## Variants

All trained with the same config (`train_shock.py`, 3000 epochs).

| Variant | How the bound is enforced | Toggle |
|---|---|---|
| **old wPINN** | none (baseline) | `USE_BOUND_PENALTY=False`, `USE_HARD_BOUNDS=False` |
| **soft penalty** | `λ·mean(relu(u−u_max)² + relu(u_min−u)²)` added to the loss | `USE_BOUND_PENALTY=True`, `LAMBDA_BOUND=10` |
| **hard bounds** | output mapped through a scaled sigmoid: `u = u_min + (u_max−u_min)·σ(z)` | `USE_HARD_BOUNDS=True` |

## Results (at `t = 0.45` unless noted)

| Variant | undershoot (min u) | overshoot (max u) | rel L1 (global) |
|---|---|---|---|
| old wPINN | **−0.3705** ❌ | +1.0120 | 0.0177 |
| soft penalty | −0.0001 ✅ | +1.0277 | 0.0274 (worse) |
| **hard bounds** | **+0.0023** ✅ | **+0.9992** ✅ | **0.0133** (best) |

`min`/`max` are over the whole space–time domain; rel L1 (global) is the
generalization error over 100k random `(t,x)` points.

See `comparison_shock_variants.png` for the slice plot at `t = 0.10 / 0.25 / 0.45`.

## Conclusion

- **Soft penalty** removes the undershoot but **smears the shock** — to avoid ever
  crossing 0 the network blurs the jump, so global accuracy *worsens* (rel L1
  0.0177 → 0.0274). The penalty competes with the data/PDE loss.
- **Hard bounds** is the clear winner: the scaled-sigmoid guarantees `u ∈ [0,1]`
  **by construction**, so there is no penalty fighting the fit. The shock stays
  sharp, the undershoot is gone, the overshoot above 1 is also fixed, and global
  accuracy is the **best of the three** (rel L1 0.0133, vs baseline 0.0177).

The original instinct — prevent out-of-bounds values — was correct; enforcing the
bound *structurally* (hard) beats penalizing it *softly*.

## Reproduce

```bash
# pick a variant in train_shock.py via USE_HARD_BOUNDS / USE_BOUND_PENALTY, then:
python3 train_shock.py            # writes ShockWave/<variant>/ModelSol.pkl
```

Models live in `ShockWave/best` (baseline), `ShockWave/bound_penalty`, and
`ShockWave/hard_bounds`.
