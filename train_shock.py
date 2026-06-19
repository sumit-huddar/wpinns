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


# ── Experiment toggles ─────────────────────────────────────────────────────
# Boundedness penalty: punish the solution network for leaving [u_min, u_max]
# (the maximum-principle band, [0, 1] for the moving shock). Set to False to
# reproduce the baseline run.
USE_BOUND_PENALTY = True
LAMBDA_BOUND = 10.0

# ── patch: override what_solving before the class is used ──────────────────
from EquationModels import ShockRarEntropy as _src

_OrigInit = _src.EquationClass.__init__

def _patched_init(self, norm, cutoff, weak_form, p):
    _OrigInit(self, norm, cutoff, weak_form, p)
    self.what_solving = "Moving"
    # Moving shock: t ∈ [0, 0.45], x ∈ [-1, 1]  (same domain as rarefaction)
    self.extrema_values = torch.tensor([[0, 0.45], [-1., 1.]])
    self.use_bound_penalty = USE_BOUND_PENALTY
    self.lambda_bound = LAMBDA_BOUND

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

output_dir = os.path.join("ShockWave", "bound_penalty" if USE_BOUND_PENALTY else "best")
os.makedirs(output_dir, exist_ok=True)

# Checkpoints (best model + full resume state) go here. Point this at a Google
# Drive folder so they survive a Colab disconnect, e.g. run with:
#   WPINN_CKPT_DIR=/content/drive/MyDrive/wpinns_out python train_shock.py
# Re-running with the same dir auto-resumes from the last saved state.
ckpt_dir = os.environ.get("WPINN_CKPT_DIR", output_dir)
os.makedirs(ckpt_dir, exist_ok=True)
checkpoint_path = os.path.join(ckpt_dir, "ModelSol.pkl")
state_path = os.path.join(ckpt_dir, "train_state.pt")
print(f"Checkpoints -> {ckpt_dir}")

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

# Move networks onto the GPU when one is available (Ec.device is already cuda
# in that case, and the fit loop moves the training tensors to it). Without
# this the model weights stay on CPU and you get a device-mismatch error.
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    solution_model.cuda()
    test_function_model.cuda()
else:
    print("No GPU found — running on CPU.")

optimizer_min = optim.Adam(solution_model.parameters(),
                           lr=ensemble_configurations["tau_sol"], amsgrad=True)
optimizer_max = optim.Adam(test_function_model.parameters(),
                           lr=ensemble_configurations["tau_test"], amsgrad=True)

# ── Train ──────────────────────────────────────────────────────────────────
print("Training wPINNs on Moving Shock …  (this takes ~4 hours on CPU)")
t0 = time.time()

best_losses, best_model, _ = fit(
    Ec,
    solution_model,
    test_function_model,
    optimizer_min,
    optimizer_max,
    dataset,
    checkpoint_path=checkpoint_path,
    checkpoint_freq=100,
    state_path=state_path,
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
