"""
Train wPINNs on a custom initial condition: a half-sine hump in the middle of
the domain, zero everywhere else —

    u0(x) = sin(pi*(x + 0.5))   for |x| <= 0.5      (peak 1 at x=0, 0 at x=+-0.5)
    u0(x) = 0                   otherwise

on x in [-1, 1], t in [0, 0.45]. Burgers has no closed-form entropy solution
for this IC, so a fine-grid Godunov run (godunov.py, case "SineBump") is used
as the ground-truth reference for evaluation.

Saves the best model to SineBump/best/ModelSol.pkl.
"""

import os
import time
import torch
import torch.optim as optim


# ── patch: solve the SineBump IC without touching the original source file ──
from EquationModels import ShockRarEntropy as _src

_OrigInit = _src.EquationClass.__init__

def _patched_init(self, norm, cutoff, weak_form, p):
    _OrigInit(self, norm, cutoff, weak_form, p)
    self.what_solving = "SineBump"
    self.extrema_values = torch.tensor([[0, 0.45], [-1., 1.]])

_src.EquationClass.__init__ = _patched_init
# ───────────────────────────────────────────────────────────────────────────

from EquationModels.ShockRarEntropy import EquationClass
from ModelClass import Pinns, PinnsTest
from FitClass import fit
from DatasetClass import DefineDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ── Hyperparameters (same config as the shock/rarefaction runs) ────────────
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

output_dir = os.path.join("SineBump", "best")
os.makedirs(output_dir, exist_ok=True)
print(f"IC: half-sine hump in [-0.5, 0.5], zero elsewhere  ->  {output_dir}")

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
print("Training wPINNs on the sine-bump IC …  (3000 epochs; ~2-3 hours on CPU)")
t0 = time.time()

best_losses, best_model, _ = fit(
    Ec,
    solution_model,
    test_function_model,
    optimizer_min,
    optimizer_max,
    dataset,
)

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed/3600:.2f} h")

# ── Evaluate (vs Godunov reference) & save ─────────────────────────────────
L2, L2_rel = Ec.compute_generalization_error(best_model, images_path=output_dir)
torch.save(best_model, os.path.join(output_dir, "ModelSol.pkl"))
print(f"Model saved -> {output_dir}/ModelSol.pkl")
print(f"L1 error = {L2:.6f},  Relative L1 = {L2_rel:.6f}  (vs Godunov reference)")

with open(os.path.join(output_dir, "InfoModel.txt"), "w") as f:
    f.write(f"what_solving, SineBump\n")
    f.write(f"train_time, {elapsed}\n")
    f.write(f"L1_vs_godunov, {L2}\n")
    f.write(f"rel_L1_vs_godunov, {L2_rel}\n")
    f.write(f"loss_tot, {best_losses[0]}\n")
