"""
Train wPINNs on the moving-shock Burgers case in one of two modes, for a fair
A/B that isolates the effect of the "optimal" recipe distilled from the
experiments in this repo:

    MODE=baseline  plain wPINN (the original method)
    MODE=optimal   plain wPINN + hard maximum-principle bounds (scaled-sigmoid
                   output, u in [u_min, u_max]) + residual-adaptive collocation
                   sampling (move points to the shock every RESAMPLE_FREQ epochs)

Everything else (architecture, seed, epochs, optimizer) is identical, so the
difference in error is attributable to the recipe.

    MODE=baseline python3 train_optimal.py   ->  ShockWave/baseline/ModelSol.pkl
    MODE=optimal  python3 train_optimal.py   ->  ShockWave/optimal/ModelSol.pkl
"""

import os
import time
import torch
import torch.optim as optim

MODE = os.environ.get("MODE", "optimal").lower()
assert MODE in ("baseline", "optimal"), "MODE must be 'baseline' or 'optimal'"
USE_HARD_BOUNDS = (MODE == "optimal")
USE_ADAPTIVE = (MODE == "optimal")
RESAMPLE_FREQ = 250
RESAMPLE_UNIFORM_FRAC = 0.5

# ── patch: solve the moving shock without touching the original source file ──
from EquationModels import ShockRarEntropy as _src

_OrigInit = _src.EquationClass.__init__

def _patched_init(self, norm, cutoff, weak_form, p):
    _OrigInit(self, norm, cutoff, weak_form, p)
    self.what_solving = "Moving"
    self.extrema_values = torch.tensor([[0, 0.45], [-1., 1.]])

_src.EquationClass.__init__ = _patched_init
# ───────────────────────────────────────────────────────────────────────────

from EquationModels.ShockRarEntropy import EquationClass
from ModelClass import Pinns, PinnsTest
from FitClass import fit
from DatasetClass import DefineDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sampling_seed = 8
N_coll = 16384
N_u    = 8192
N_int  = 0

ensemble_configurations = {
    "hidden_layers_sol":  6,
    "hidden_layers_test": 4,
    "neurons_sol":  20,
    "neurons_test": 10,
    "activation_sol":  "tanh",
    "activation_test": "tanh",
    "tau_sol":  0.01,
    "tau_test": 0.015,
    "iterations_min": 1,
    "iterations_max": 8,
    "residual_parameter": 10,
    "kernel_regularizer": 2,
    "regularization_parameter_sol":  0.,
    "regularization_parameter_test": 0.,
    "batch_size": N_coll + N_u + N_int,
    "epochs": 3000,
    "norm": "H1",
    "cutoff": "def_max",
    "weak_form": "partial",
    "reset_freq": 0.025,
    "loss_type": "l2",
}

output_dir = os.path.join("ShockWave", MODE)
os.makedirs(output_dir, exist_ok=True)
print(f"MODE = {MODE}  (hard_bounds={USE_HARD_BOUNDS}, adaptive={USE_ADAPTIVE})  ->  {output_dir}")

Ec = EquationClass(
    norm=ensemble_configurations["norm"],
    cutoff=ensemble_configurations["cutoff"],
    weak_form=ensemble_configurations["weak_form"],
    p=2,
)

N_u_train = N_u
N_b_train = int(N_u_train / (4 * Ec.space_dimensions))
N_i_train = N_u_train - 2 * Ec.space_dimensions * N_b_train

dataset = DefineDataset(Ec, N_coll, N_b_train, N_i_train, N_int,
                        ensemble_configurations["batch_size"],
                        sampling_seed,
                        shuffle=False)
dataset.assemble_dataset()

network_props_sol = {
    "hidden_layers":          ensemble_configurations["hidden_layers_sol"],
    "neurons":                ensemble_configurations["neurons_sol"],
    "residual_parameter":     ensemble_configurations["residual_parameter"],
    "kernel_regularizer":     ensemble_configurations["kernel_regularizer"],
    "regularization_parameter": ensemble_configurations["regularization_parameter_sol"],
    "epochs":                 ensemble_configurations["epochs"],
    "activation":             ensemble_configurations["activation_sol"],
    "iterations":             ensemble_configurations["iterations_max"],
    "reset_freq":             ensemble_configurations["reset_freq"],
    "loss_type":              ensemble_configurations["loss_type"],
}

network_props_test = {
    "hidden_layers":          ensemble_configurations["hidden_layers_test"],
    "neurons":                ensemble_configurations["neurons_test"],
    "residual_parameter":     ensemble_configurations["residual_parameter"],
    "kernel_regularizer":     ensemble_configurations["kernel_regularizer"],
    "regularization_parameter": ensemble_configurations["regularization_parameter_test"],
    "epochs":                 ensemble_configurations["epochs"],
    "activation":             ensemble_configurations["activation_test"],
    "iterations":             ensemble_configurations["iterations_max"],
    "reset_freq":             ensemble_configurations["reset_freq"],
}

torch.manual_seed(42)
solution_model = Pinns(Ec.space_dimensions + Ec.time_dimensions, Ec.output_dimension, network_props_sol)
test_function_model = PinnsTest(Ec.space_dimensions + Ec.time_dimensions, Ec.output_dimension, network_props_test)

# OPTIMAL ingredient 1: hard maximum-principle bounds on the solution network
if USE_HARD_BOUNDS:
    lo, hi = Ec.bounds()
    solution_model.set_output_bounds(lo, hi)
    print(f"Hard output bounds: u in [{float(lo):.3g}, {float(hi):.3g}]")

optimizer_min = optim.Adam(solution_model.parameters(), lr=ensemble_configurations["tau_sol"], amsgrad=True)
optimizer_max = optim.Adam(test_function_model.parameters(), lr=ensemble_configurations["tau_test"], amsgrad=True)

print(f"Training wPINNs on Moving Shock ({MODE}) … (3000 epochs; ~30-60 min on CPU)")
t0 = time.time()

# OPTIMAL ingredient 2: residual-adaptive collocation sampling
best_losses, best_model, _ = fit(
    Ec, solution_model, test_function_model, optimizer_min, optimizer_max, dataset,
    resample_freq=(RESAMPLE_FREQ if USE_ADAPTIVE else 0),
    resample_uniform_frac=RESAMPLE_UNIFORM_FRAC,
)

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed/3600:.2f} h")

L2, L2_rel = Ec.compute_generalization_error(best_model, images_path=output_dir)
torch.save(best_model, os.path.join(output_dir, "ModelSol.pkl"))
print(f"Model saved -> {output_dir}/ModelSol.pkl")
print(f"L1 error = {L2:.6f},  Relative L1 = {L2_rel:.6f}")

with open(os.path.join(output_dir, "InfoModel.txt"), "w") as f:
    f.write(f"mode, {MODE}\n")
    f.write(f"hard_bounds, {USE_HARD_BOUNDS}\n")
    f.write(f"adaptive, {USE_ADAPTIVE}\n")
    f.write(f"train_time, {elapsed}\n")
    f.write(f"L1, {L2}\n")
    f.write(f"rel_L1, {L2_rel}\n")
    f.write(f"loss_tot, {best_losses[0]}\n")
