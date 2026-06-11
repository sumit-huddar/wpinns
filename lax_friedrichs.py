"""
Lax-Friedrichs solver for inviscid Burgers' equation  u_t + (u^2/2)_x = 0
on x ∈ [x_min, x_max], t ∈ [0, T].

Two Riemann problems (matching the wPINNs experiments):
  - "Rarefaction": u0 = -1 (x≤0), +1 (x>0)  →  rarefaction fan
  - "Moving":      u0 = +1 (x≤0),  0 (x>0)  →  rightward-moving shock at speed 1/2
"""

import numpy as np


def lax_friedrichs(case: str, nx: int = 500, cfl: float = 0.9):
    """
    Run the Lax-Friedrichs scheme and return the solution on a (t, x) grid.

    Parameters
    ----------
    case : "Rarefaction" or "Moving"
    nx   : number of interior spatial cells
    cfl  : CFL number (must be < 1 for stability)

    Returns
    -------
    x_centers : (nx,)   spatial cell centers
    t_vec     : (nt,)   time values at which solution is stored
    U         : (nt, nx) solution array
    """
    if case == "Rarefaction":
        x_min, x_max, T = -1.0, 1.0, 0.45
    elif case == "Moving":
        x_min, x_max, T = -1.0, 1.0, 0.45
    else:
        raise ValueError(f"Unknown case '{case}'. Use 'Rarefaction' or 'Moving'.")

    dx = (x_max - x_min) / nx
    x_centers = np.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, nx)

    # Initial condition
    if case == "Rarefaction":
        u = np.where(x_centers <= 0.0, -1.0, 1.0).astype(float)
        u_left_bc, u_right_bc = -1.0, 1.0
    else:  # Moving shock
        u = np.where(x_centers <= 0.0, 1.0, 0.0).astype(float)
        u_left_bc, u_right_bc = 1.0, 0.0

    def flux(v):
        return 0.5 * v ** 2

    t = 0.0
    t_vec = [t]
    U = [u.copy()]

    while t < T:
        max_wave = max(np.max(np.abs(u)), 1e-12)
        dt = cfl * dx / max_wave
        dt = min(dt, T - t)

        # Ghost cells for Dirichlet BCs
        u_ext = np.concatenate([[u_left_bc], u, [u_right_bc]])

        f = flux(u_ext)
        # Lax-Friedrichs flux: F_{j+1/2} = (f_j + f_{j+1})/2 - dx/(2dt)*(u_{j+1} - u_j)
        F = 0.5 * (f[:-1] + f[1:]) - (dx / (2.0 * dt)) * (u_ext[1:] - u_ext[:-1])

        u_new = u - (dt / dx) * (F[1:] - F[:-1])

        u = u_new
        t += dt
        t_vec.append(t)
        U.append(u.copy())

    return x_centers, np.array(t_vec), np.array(U)


def exact_solution(case: str, x: np.ndarray, t: float) -> np.ndarray:
    """
    Exact entropy solution evaluated at spatial positions x and time t.

    Rarefaction (u0 = -1 left, +1 right):
        u = -1          x < -t
        u = x/t    -t ≤ x ≤ t
        u = +1          x > t

    Moving shock (u0 = 1 left, 0 right, shock speed s = 1/2):
        u = 1      x < t/2
        u = 0      x > t/2
    """
    sol = np.zeros_like(x, dtype=float)
    if case == "Rarefaction":
        if t == 0.0:
            sol = np.where(x <= 0.0, -1.0, 1.0).astype(float)
        else:
            sol = np.where(x < -t, -1.0,
                  np.where(x > t,   1.0, x / t))
    elif case == "Moving":
        if t == 0.0:
            sol = np.where(x <= 0.0, 1.0, 0.0).astype(float)
        else:
            sol = np.where(x <= 0.5 * t, 1.0, 0.0).astype(float)
    return sol


def l1_error(u_num: np.ndarray, u_ex: np.ndarray) -> float:
    return float(np.mean(np.abs(u_num - u_ex)))


def relative_l1_error(u_num: np.ndarray, u_ex: np.ndarray) -> float:
    denom = np.mean(np.abs(u_ex))
    return l1_error(u_num, u_ex) / max(denom, 1e-12)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for case in ["Rarefaction", "Moving"]:
        x, t_vec, U = lax_friedrichs(case, nx=500)
        T_final = t_vec[-1]
        u_lf = U[-1]
        u_ex = exact_solution(case, x, T_final)

        print(f"\n=== {case} ===")
        print(f"  Final time : {T_final:.4f}")
        print(f"  L1 error   : {l1_error(u_lf, u_ex):.6f}")
        print(f"  Rel L1 err : {relative_l1_error(u_lf, u_ex):.6f}")

        plt.figure(figsize=(7, 4))
        plt.plot(x, u_ex, "k-",  lw=2,   label="Exact")
        plt.plot(x, u_lf, "r--", lw=1.5, label="Lax-Friedrichs")
        plt.title(f"Burgers — {case}  (t = {T_final:.2f})")
        plt.xlabel("x"); plt.ylabel("u")
        plt.legend(); plt.grid(True, ls=":")
        plt.tight_layout()
        plt.savefig(f"lf_{case.lower()}.png", dpi=150)
        plt.show()
