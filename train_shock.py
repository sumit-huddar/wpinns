"""
Train wPINNs on the moving-shock Burgers case and save the best model to
ShockWave/best/ModelSol.pkl  (the path comparison.ipynb expects).

Usage:
    python train_shock.py

This script patches ShockRarEntropy.EquationClass so that what_solving is
set to "Moving" without touching the original source file.
"""

import sys
import os
import json
import copy
import time
import torch
import torch.optim as optim


# ── Case: "Rarefaction" (fan) or "Moving" (right-moving shock) ──────────────
# Override at run time with the WPINN_CASE env var.
CASE = os.environ.get("WPINN_CASE", "Rarefaction")

# ── patch: override what_solving before the class is used ──────────────────
from EquationModels import ShockRarEntropy as _src

_OrigInit = _src.EquationClass.__init__

def _patched_init(self, norm, cutoff, weak_form, p):
    _OrigInit(self, norm, cutoff, weak_form, p)
    self.what_solving = CASE
    # t ∈ [0, 0.45], x ∈ [-1, 1]
    self.extrema_values = torch.tensor([[0, 0.45], [-1., 1.]])

_src.EquationClass.__init__ = _patched_init
# ───────────────────────────────────────────────────────────────────────────

from EquationModels.ShockRarEntropy import EquationClass
from ModelClass import Pinns, PinnsTest
from FitClass import fit
from DatasetClass import DefineDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ── Hyperparameters (best config from RarefactionWave/best.csv) ────────────
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

# ── Experiment: uniform vs residual-adaptive collocation sampling ───────────
# Both runs use identical config; only the collocation sampling differs.
#   USE_ADAPTIVE_SAMPLING=False -> original solver, uniform points
#   USE_ADAPTIVE_SAMPLING=True  -> every RESAMPLE_FREQ epochs, move points
#                                  toward the highest-residual region (shock)
# Override USE_ADAPTIVE_SAMPLING at run time with WPINN_ADAPTIVE=0/1.
USE_ADAPTIVE_SAMPLING = os.environ.get("WPINN_ADAPTIVE", "1") == "1"
RESAMPLE_FREQ = 250
RESAMPLE_UNIFORM_FRAC = 0.5

variant = "adaptive" if USE_ADAPTIVE_SAMPLING else "uniform"
base_dir = "RarefactionWave" if CASE == "Rarefaction" else "ShockWave"
output_dir = os.path.join(base_dir, variant)
os.makedirs(output_dir, exist_ok=True)
print(f"Case: {CASE}   Variant: {variant}  ->  {output_dir}")

# ── Build equation / dataset ───────────────────────────────────────────────
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

# ── Build networks ─────────────────────────────────────────────────────────
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
solution_model = Pinns(
    input_dimension=Ec.space_dimensions + Ec.time_dimensions,
    output_dimension=Ec.output_dimension,
    network_properties=network_props_sol,
)

test_function_model = PinnsTest(
    input_dimension=Ec.space_dimensions + Ec.time_dimensions,
    output_dimension=Ec.output_dimension,
    network_properties=network_props_test,
)

optimizer_min = optim.Adam(solution_model.parameters(),
                           lr=ensemble_configurations["tau_sol"], amsgrad=True)
optimizer_max = optim.Adam(test_function_model.parameters(),
                           lr=ensemble_configurations["tau_test"], amsgrad=True)

# ── Train ──────────────────────────────────────────────────────────────────
print(f"Training wPINNs on {CASE} …  (3000 epochs; ~2-3 hours on CPU)")
t0 = time.time()

best_losses, best_model, _ = fit(
    Ec,
    solution_model,
    test_function_model,
    optimizer_min,
    optimizer_max,
    dataset,
    resample_freq=(RESAMPLE_FREQ if USE_ADAPTIVE_SAMPLING else 0),
    resample_uniform_frac=RESAMPLE_UNIFORM_FRAC,
)

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed/3600:.2f} h")

# ── Evaluate & save ────────────────────────────────────────────────────────
L2, L2_rel = Ec.compute_generalization_error(best_model, images_path=output_dir)
torch.save(best_model, os.path.join(output_dir, "ModelSol.pkl"))
print(f"Model saved → {output_dir}/ModelSol.pkl")
print(f"L1 error = {L2:.6f},  Relative L1 = {L2_rel:.6f}")

with open(os.path.join(output_dir, "InfoModel.txt"), "w") as f:
    f.write(f"what_solving, {CASE}\n")
    f.write(f"train_time, {elapsed}\n")
    f.write(f"L2_norm_test, {L2}\n")
    f.write(f"rel_L2_norm, {L2_rel}\n")
    f.write(f"loss_tot, {best_losses[0]}\n")
