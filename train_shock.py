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

# ── patch: override what_solving before the class is used ──────────────────
from EquationModels import ShockRarEntropy as _src

_OrigInit = _src.EquationClass.__init__

def _patched_init(self, norm, cutoff, weak_form, p):
    _OrigInit(self, norm, cutoff, weak_form, p)
    self.what_solving = "Moving"
    # Moving shock: t ∈ [0, 0.45], x ∈ [-1, 1]  (same domain as rarefaction)
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
    "epochs": 5000,
    "norm": "H1",
    "cutoff": "def_max",
    "weak_form": "partial",
    "reset_freq": 0.025,
    "loss_type": "l2",
}

output_dir = os.path.join("ShockWave", "best")
os.makedirs(output_dir, exist_ok=True)

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
                        sampling_seed, ensemble_configurations["batch_size"],
                        ensemble_configurations["batch_size"],
                        shuffle=False)

# ── Build networks ─────────────────────────────────────────────────────────
loss_fn = torch.nn.MSELoss()

torch.manual_seed(42)
solution_model = Pinns(
    input_dimension=Ec.space_dimensions + Ec.time_dimensions,
    output_dimension=Ec.output_dimension,
    n_hidden_layers=ensemble_configurations["hidden_layers_sol"],
    neurons=ensemble_configurations["neurons_sol"],
    regularization_param=ensemble_configurations["regularization_parameter_sol"],
    regularization_exp=ensemble_configurations["kernel_regularizer"],
    retrain_seed=42,
    activation_name=ensemble_configurations["activation_sol"],
    loss=loss_fn,
    n_iterations_min=ensemble_configurations["iterations_min"],
    n_iterations_max=ensemble_configurations["iterations_max"],
    reset_freq=ensemble_configurations["reset_freq"],
)

test_function_model = PinnsTest(
    input_dimension=Ec.space_dimensions + Ec.time_dimensions,
    output_dimension=Ec.output_dimension,
    n_hidden_layers=ensemble_configurations["hidden_layers_test"],
    neurons=ensemble_configurations["neurons_test"],
    regularization_param=ensemble_configurations["regularization_parameter_test"],
    regularization_exp=ensemble_configurations["kernel_regularizer"],
    retrain_seed=42,
    activation_name=ensemble_configurations["activation_test"],
    loss=loss_fn,
    n_iterations_min=ensemble_configurations["iterations_min"],
    n_iterations_max=ensemble_configurations["iterations_max"],
    reset_freq=ensemble_configurations["reset_freq"],
)

optimizer_min = optim.Adam(solution_model.parameters(),
                           lr=ensemble_configurations["tau_sol"], amsgrad=True)
optimizer_max = optim.Adam(test_function_model.parameters(),
                           lr=ensemble_configurations["tau_test"], amsgrad=True)

# ── Train ──────────────────────────────────────────────────────────────────
print("Training wPINNs on Moving Shock …  (this takes ~4 hours on CPU)")
t0 = time.time()

best_model, best_losses = fit(
    Ec,
    solution_model,
    test_function_model,
    dataset.data_coll,
    dataset.data_boundary,
    dataset.data_initial_internal,
    optimizer_min,
    optimizer_max,
    ensemble_configurations["epochs"],
    ensemble_configurations["residual_parameter"],
    output_dir,
)

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed/3600:.2f} h")

# ── Evaluate & save ────────────────────────────────────────────────────────
L2, L2_rel = Ec.compute_generalization_error(best_model, images_path=output_dir)
torch.save(best_model, os.path.join(output_dir, "ModelSol.pkl"))
print(f"Model saved → {output_dir}/ModelSol.pkl")
print(f"L1 error = {L2:.6f},  Relative L1 = {L2_rel:.6f}")

with open(os.path.join(output_dir, "InfoModel.txt"), "w") as f:
    f.write(f"what_solving, Moving\n")
    f.write(f"train_time, {elapsed}\n")
    f.write(f"L2_norm_test, {L2}\n")
    f.write(f"rel_L2_norm, {L2_rel}\n")
    f.write(f"loss_tot, {best_losses[0]}\n")
