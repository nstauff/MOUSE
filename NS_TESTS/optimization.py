# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""
Multi-objective Bayesian optimization for GCMR Design A.
Minimizes LCOE (FOAK and NOAK) subject to safety constraints (temp. coeff., shutdown margin).
"""

import contextlib
import multiprocessing as mp
import os
import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt
from .def_watts_exec_GCMR_Design_A import gcmr_calc


@contextlib.contextmanager
def redirect_all_output(log_file):
    """Redirect Python-level and C-level stdout/stderr to a log file.
    contextlib.redirect_stdout only hooks sys.stdout; OpenMC writes at the
    OS file-descriptor level, so we must also redirect fd 1 and fd 2 with
    os.dup2 to suppress those messages from the terminal.
    """
    with open(log_file, 'w') as f:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = f
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

import torch
import time
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from botorch.models import SingleTaskGP
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

# Use double precision throughout (BoTorch best practice)
dtype = torch.float64
# --------------------------------------------------------------------------- #
# Parallelism configuration
# --------------------------------------------------------------------------- #
N_WORKERS = 4  # number of parallel workers per batch
BATCH_SIZE = 4  # it is much more efficient to have only 1 worker per batch!
USE_PROCESSES = True
n_initial_points = 2**BATCH_SIZE
extended_penalty = True
# --------------------------------------------------------------------------- #
# Integer variable configuration
# --------------------------------------------------------------------------- #
INTEGER_INDICES = [4]  # Active Height is treated as integer (cm); all others are continuous
# CSV column headers  — order must match: params (5) + objectives (2) + constraints (3)
CSV_HEADER = [
    "Enrichment", "Packing Fraction", "Compact Fuel Radius", "Moderator Booster Radius", "Active Height",
    "LCOE FOAK", "LCOE NOAK",
    "Temp Coeff (pcm/K)", "-SD Margin (pcm)", "Peak Factor - 2.0",
]
# --------------------------------------------------------------------------- #
# Problem bounds in ORIGINAL (raw) space
# --------------------------------------------------------------------------- #
raw_bounds = torch.tensor(
    [
        [0.10, 0.1975],  # Enrichment
        [0.15, 0.40],      # Packing Fraction
        [0.40, 0.6225],   # Compact Fuel Radius
        [0.40, 0.6225],   # Moderator Booster Radius
        [150, 300], # Active Height
    ],
    dtype=dtype,
).T  # shape (2, 5)
# Unit cube bounds for the GP / acquisition function
unit_bounds = torch.zeros(2, raw_bounds.shape[1], dtype=dtype)
unit_bounds[1] = 1.0
# --------------------------------------------------------------------------- #
# Normalization / unnormalization helpers
# --------------------------------------------------------------------------- #
def normalize_to_unit(x_raw):
    return normalize(x_raw, bounds=raw_bounds)
def unnormalize_to_raw(x_unit):
    return unnormalize(x_unit, bounds=raw_bounds)
# --------------------------------------------------------------------------- #
# Rounding helper (operates in RAW space)
# --------------------------------------------------------------------------- #
def round_to_integers_raw(x_raw, integer_indices):
    x_rounded = x_raw.clone()
    for idx in integer_indices:
        x_rounded[..., idx] = torch.round(x_rounded[..., idx])
        x_rounded[..., idx] = torch.clamp(
            x_rounded[..., idx],
            min=raw_bounds[0, idx].item(),
            max=raw_bounds[1, idx].item(),
        )
    return x_rounded
# --------------------------------------------------------------------------- #
# Objective evaluation
# --------------------------------------------------------------------------- #
def evaluate_objective(x_list, output_dir):
    """
    Parameters
    ----------
    x_list     : list of floats in RAW space (already rounded to integers)
    output_dir : directory where log and Excel files for this run are written
    Returns
    -------
    dict with keys:
        "params"     : list of 5 floats [Enrichment, Packing Fraction, Compact Fuel Radius, Moderator Booster Radius, Active Height]
        "objectives" : list of 2 floats [LCOE_FOAK, LCOE_NOAK] WITHOUT penalty
        "constraints": list of 3 floats [g1=temp_coeff (<=0 desired), g2=-SD_margin (<=0 desired), g3=peak_factor-2.0 (<=0 desired)]
        "penalised"  : list of 2 floats [f1+penalty, f2+penalty] WITH penalty (used by BO)
    """

    varied_params = {
        'Enrichment': x_list[0],
        'Moderator Booster':   'ZrH',
        'Packing Fraction':    x_list[1],
        'Compact Fuel Radius': x_list[2],
        'Moderator Booster Radius': x_list[3],
        'Active Height': x_list[4],
    }
    calc_id = f"opt_E{x_list[0]:.4f}_PF{x_list[1]:.3f}_R{x_list[2]:.4f}_MB{x_list[3]:.4f}_AH{x_list[4]:.0f}"
    log_file = os.path.join(output_dir, f"log_{calc_id}.txt")

    print(f"Running {calc_id} (log -> {log_file})")

    with redirect_all_output(log_file):
        results = gcmr_calc(varied_params, calc_id, output_dir)
    temp_coeff, sd_margin, fuel_lifetime, max_peak_factor, lcoe_foak_val, lcoe_noak_val = results
    print(
        f"Temp Coeff: {temp_coeff:.2f} pcm/K | "
        f"SD Margin: {sd_margin:.0f} pcm | "
        f"Fuel Lifetime: {fuel_lifetime:.1f} days | "
        f"Max Peak Factor: {max_peak_factor:.3f} | "
        f"LCOE FOAK: {lcoe_foak_val:.2f} $/MWh | "
        f"LCOE NOAK: {lcoe_noak_val:.2f} $/MWh"
    )

    f1 = lcoe_foak_val
    f2 = lcoe_noak_val
    g1 = temp_coeff
    g2 = sd_margin
    g3 = max_peak_factor - 2.0
    penalty = 0.0
    if g1 > 0:
        penalty += abs(g1) * 1000
    if g2 > 0:
        penalty += abs(g2) 
    if g3 > 0:
        penalty += abs(g3) * 1000

    return {
        "params": x_list,
        "objectives": [f1, f2],
        "constraints": [g1, g2, g3],
        "penalised": [f1 + penalty, f2 + penalty],
    }
# --------------------------------------------------------------------------- #
# Parallel batch evaluation
# --------------------------------------------------------------------------- #
def _fallback_result(x_list):
    return {
        "params": x_list,
        "objectives": [1e6, 1e6],
        "constraints": [1e6, 1e6, 1e6],
        "penalised": [1e6, 1e6],
    }

def evaluate_batch_parallel(x_batch_raw, output_dir, n_workers=N_WORKERS, use_processes=USE_PROCESSES):
    x_lists = [row.tolist() for row in x_batch_raw]
    results = [None] * len(x_lists)

    if n_workers == 1:
        # Sequential path: avoids fork+OpenMP deadlocks; multiprocessing can be
        # re-enabled for N_WORKERS > 1 (uses 'spawn' context, which is safe with OpenMC).
        for i, x in enumerate(x_lists):
            try:
                results[i] = evaluate_objective(x, output_dir)
            except Exception as exc:
                print(f"  [ERROR] Evaluation {i} raised: {exc}")
                traceback.print_exc()
                results[i] = _fallback_result(x_lists[i])
    else:
        # Parallel path: 'spawn' avoids the fork+OpenMP deadlock that the default
        # 'fork' start method causes when OpenMC/OpenMP threads are active.
        PoolExecutor = (
            lambda **kw: ProcessPoolExecutor(mp_context=mp.get_context("spawn"), **kw)
            if use_processes else ThreadPoolExecutor
        )
        with PoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(evaluate_objective, x, output_dir): i
                for i, x in enumerate(x_lists)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"  [ERROR] Evaluation {idx} raised: {exc}")
                    traceback.print_exc()
                    results[idx] = _fallback_result(x_lists[idx])

    penalised_y = torch.tensor(
        [r["penalised"] for r in results], dtype=dtype
    )
    return penalised_y, results
# --------------------------------------------------------------------------- #
# CSV writing helpers
# --------------------------------------------------------------------------- #
def write_csv_header(filepath):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
def append_results_to_csv(filepath, results_list):
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        for r in results_list:
            enr, pf, cfr, mbr, ah = r["params"]
            row = (
                [f"{enr:.4f}", f"{pf:.3f}", f"{cfr:.4f}", f"{mbr:.4f}", f"{ah:.0f}"]
                + [f"{o:.6f}" for o in r["objectives"]]
                + [f"{c:.6f}" for c in r["constraints"]]
            )
            writer.writerow(row)
def write_pareto_csv(filepath, pareto_x_raw, all_results):
    def _param_key(params):
        enr, pf, cfr, mbr, ah = params
        return (round(enr, 4), round(pf, 3), round(cfr, 4), round(mbr, 4), round(ah, 0))

    results_lookup = {_param_key(r["params"]): r for r in all_results}
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in pareto_x_raw:
            key = _param_key(row.tolist())
            r = results_lookup.get(key)
            if r is not None:
                enr, pf, cfr, mbr, ah = r["params"]
                csv_row = (
                    [f"{enr:.4f}", f"{pf:.3f}", f"{cfr:.4f}", f"{mbr:.4f}", f"{ah:.0f}"]
                    + [f"{o:.6f}" for o in r["objectives"]]
                    + [f"{c:.6f}" for c in r["constraints"]]
                )
            else:
                enr, pf, cfr, mbr, ah = row.tolist()
                csv_row = [f"{enr:.4f}", f"{pf:.3f}", f"{cfr:.4f}", f"{mbr:.4f}", f"{ah:.0f}"]
                csv_row += ["N/A"] * 5
            writer.writerow(csv_row)
# --------------------------------------------------------------------------- #
# Deduplication (operates in RAW space)
# --------------------------------------------------------------------------- #
def deduplicate(candidates_raw, train_x_raw):
    mask = []
    for i in range(candidates_raw.shape[0]):
        is_dup = (train_x_raw == candidates_raw[i]).all(dim=1).any()
        mask.append(not is_dup)
    mask = torch.tensor(mask)
    if mask.sum() == 0:
        print("  [WARN] All candidates are duplicates — keeping first anyway")
        mask[0] = True
    return candidates_raw[mask]
# --------------------------------------------------------------------------- #
# Build model + acquisition function (all in UNIT space)
# --------------------------------------------------------------------------- #
def build_model_and_acqf(train_x_unit, train_y, ref_point, batch_size=BATCH_SIZE):
    gp = SingleTaskGP(train_x_unit, train_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_y)
    acq_func = qLogExpectedHypervolumeImprovement(
        model=gp,
        ref_point=ref_point,
        partitioning=partitioning,
    )
    return gp, acq_func
# =========================================================================== #
#  MAIN — everything that should run ONLY ONCE goes here
# =========================================================================== #
if __name__ == "__main__":
    start_time = time.time()
    # ---- Create output directory ONCE, inside the main guard ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    algo = "botorch"
    output_dir = os.path.join("NS_TESTS", f"{algo}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    ALL_EVALS_FILE = os.path.join(output_dir, "all_evaluations.csv")
    PARETO_FILE = os.path.join(output_dir, "pareto_front.csv")
    # Initialise the all-evaluations CSV
    write_csv_header(ALL_EVALS_FILE)
    # Collect ALL evaluation results for Pareto lookup at the end
    all_eval_results = []
    # ------------------------------------------------------------------- #
    # Generate initial training data
    # ------------------------------------------------------------------- #
    train_x_unit = draw_sobol_samples(
        bounds=unit_bounds, n=n_initial_points, q=1
    ).squeeze(1).to(dtype=dtype)
    train_x_raw = unnormalize_to_raw(train_x_unit)
    train_x_raw = round_to_integers_raw(train_x_raw, INTEGER_INDICES)
    train_x_unit = normalize_to_unit(train_x_raw)

    print(f"\nEvaluating {n_initial_points} initial points "
        f"with {N_WORKERS} workers …")
    t0 = time.time()
    train_y_pen, init_results = evaluate_batch_parallel(train_x_raw, output_dir)
    print(f"  Done in {time.time() - t0:.2f}s")
    append_results_to_csv(ALL_EVALS_FILE, init_results)
    all_eval_results.extend(init_results)
    train_y = -train_y_pen  # negate for maximisation
    ref_point = torch.tensor([-500.0, -10.0], dtype=dtype)
    # ------------------------------------------------------------------- #
    # Bayesian optimisation loop
    # ------------------------------------------------------------------- #
    n_iterations = 20
    total_evals = n_initial_points
    gp, acq_func = build_model_and_acqf(train_x_unit, train_y, ref_point)
    for i in range(n_iterations):
        iter_t0 = time.time()
        candidates_unit, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=unit_bounds,
            q=BATCH_SIZE,
            num_restarts=10,
            raw_samples=256,
        )
        candidates_unit = candidates_unit.view(-1, raw_bounds.shape[1])
        candidates_raw = unnormalize_to_raw(candidates_unit)
        candidates_raw = round_to_integers_raw(candidates_raw, INTEGER_INDICES)
        candidates_raw = deduplicate(candidates_raw, train_x_raw)
        candidates_unit_rounded = normalize_to_unit(candidates_raw)
        new_y_pen, iter_results = evaluate_batch_parallel(candidates_raw, output_dir)
        new_y = -new_y_pen
        append_results_to_csv(ALL_EVALS_FILE, iter_results)
        all_eval_results.extend(iter_results)
        train_x_raw = torch.cat([train_x_raw, candidates_raw], dim=0)
        train_x_unit = torch.cat([train_x_unit, candidates_unit_rounded], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)
        total_evals += candidates_raw.shape[0]
        gp, acq_func = build_model_and_acqf(train_x_unit, train_y, ref_point)
        elapsed = time.time() - iter_t0
        n_pareto = is_non_dominated(train_y).sum().item()
        print(
            f"Iteration {i + 1:3d}/{n_iterations} | "
            f"batch q={candidates_raw.shape[0]} | "
            f"total evals={total_evals} | "
            f"Pareto size={n_pareto} | "
            f"time={elapsed:.2f}s"
        )
    # ------------------------------------------------------------------- #
    # Pareto refinement phase — fill gaps in the front
    # ------------------------------------------------------------------- #
    n_refine_iterations = 10
    REFINE_BATCH_SIZE = BATCH_SIZE
    print("\n" + "=" * 60)
    print("Starting Pareto refinement phase …")
    print("=" * 60)
    for i in range(n_refine_iterations):
        iter_t0 = time.time()
        # Rebuild model with latest data
        gp_refine, acq_refine = build_model_and_acqf(
            train_x_unit, train_y, ref_point, batch_size=REFINE_BATCH_SIZE
        )
        # Use more restarts and raw samples for better coverage
        candidates_unit, acq_value = optimize_acqf(
            acq_function=acq_refine,
            bounds=unit_bounds,
            q=REFINE_BATCH_SIZE,
            num_restarts=20,       # more restarts for diversity
            raw_samples=512,       # more raw samples
        )
        candidates_unit = candidates_unit.view(-1, raw_bounds.shape[1])
        candidates_raw = unnormalize_to_raw(candidates_unit)
        candidates_raw = round_to_integers_raw(candidates_raw, INTEGER_INDICES)
        candidates_raw = deduplicate(candidates_raw, train_x_raw)
        if candidates_raw.shape[0] == 0:
            print(f"  Refine {i + 1}: no new candidates, skipping")
            continue
        candidates_unit_rounded = normalize_to_unit(candidates_raw)
        new_y_pen, iter_results = evaluate_batch_parallel(candidates_raw, output_dir)
        new_y = -new_y_pen
        append_results_to_csv(ALL_EVALS_FILE, iter_results)
        all_eval_results.extend(iter_results)
        train_x_raw = torch.cat([train_x_raw, candidates_raw], dim=0)
        train_x_unit = torch.cat([train_x_unit, candidates_unit_rounded], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)
        total_evals += candidates_raw.shape[0]
        elapsed = time.time() - iter_t0
        n_pareto = is_non_dominated(train_y).sum().item()
        print(
            f"  Refine {i + 1:3d}/{n_refine_iterations} | "
            f"batch q={candidates_raw.shape[0]} | "
            f"total evals={total_evals} | "
            f"Pareto size={n_pareto} | "
            f"time={elapsed:.2f}s"
        )
    # ------------------------------------------------------------------- #
    # Extract Pareto front and write to file
    # ------------------------------------------------------------------- #
    all_y_raw = -train_y
    pareto_mask = is_non_dominated(train_y)
    pareto_front = all_y_raw[pareto_mask]
    pareto_x = train_x_raw[pareto_mask]
    write_pareto_csv(PARETO_FILE, pareto_x, all_eval_results)
    # ------------------------------------------------------------------- #
    # Console summary
    # ------------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("Pareto Front (raw objectives — lower is better):")
    print("=" * 60)
    print(f"{'LCOE FOAK':>12s}  {'LCOE NOAK':>12s}")
    for row in pareto_front:
        print(f"{row[0].item():12.4f}  {row[1].item():12.4f}")
    print("\nCorresponding input parameters (integer values):")
    print(f"{'Enrichment':>12s}  {'Pack. Frac.':>12s}  {'Fuel Radius':>12s}  {'Boost. Rad.':>12s}  {'Act. Height':>12s}")
    for row in pareto_x:
        print(
            f"{row[0].item():12.4f}  {row[1].item():12.3f}  "
            f"{row[2].item():12.4f}  {row[3].item():12.4f}  {row[4].item():12.0f}"
        )
    print(f"\nTotal evaluations: {total_evals}")
    print(f"All evaluations written to: {ALL_EVALS_FILE}")
    print(f"Pareto front written to:    {PARETO_FILE}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n===== Optimization Summary =====")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Total time: {elapsed_time/60:.2f} minutes")