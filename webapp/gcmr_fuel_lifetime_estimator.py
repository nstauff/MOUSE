# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
GCMR fuel-lifetime estimator — local KNN regression.

Model
-----
    Lifetime = F_A × F_C × H × A1 × (Enrichment − A2) / Power

where
    F_A = 3(N_A−1)² − 3(N_A−1) + 1   (fuel compacts per assembly)
    F_C = 3 N_C²  − 3 N_C  + 1        (fuel assemblies in core)

For each query the K=4 nearest training points are found in the
(Enrichment, N_A, N_C, H, Power) feature space (raw features normalised
to [0,1] for the distance calculation).  A1 and A2 are then fitted by
simple linear regression of the normalised lifetime

    L* = Lifetime × Power / (F_A × F_C × H)

against Enrichment on those K neighbours.  Subcritical cases have
L*=0 and are included in the training pool — they anchor the local
fit near the criticality boundary and improve A2 estimation.

Accuracy (LOO-CV on 132 non-zero training points)
--------------------------------------------------
  Median absolute error : ~129 days
  Median MAPE           : ~5.9 %

Reference data
--------------
  assets/Ref_Results/GCMR_parametric_size_power_enrichment.xlsx
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data path and module-level cache
# ---------------------------------------------------------------------------
_XLSX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'assets', 'Ref_Results', 'GCMR_parametric_size_power_enrichment.xlsx',
)

_K = 4          # nearest neighbours to use for local fit

_train_df = None    # cached training data (loaded once)
_feat_min = None    # min of each feature — used for [0,1] normalisation
_feat_max = None    # max of each feature


def _assembly_factor(n_a):
    """Fuel compacts per assembly: 3(N_A−1)²−3(N_A−1)+1."""
    m = int(n_a) - 1
    return 3 * m * m - 3 * m + 1


def _core_factor(n_c):
    """Fuel assemblies in core: 3N_C²−3N_C+1."""
    n = int(n_c)
    return 3 * n * n - 3 * n + 1


def _load():
    """Load and cache the training data on first call.

    All rows including subcritical (LT=0) are loaded — they anchor the
    local linear fit near the criticality boundary.
    """
    global _train_df, _feat_min, _feat_max
    if _train_df is not None:
        return

    df = pd.read_excel(_XLSX_PATH, sheet_name='Merged Data', engine='openpyxl')
    df = df[['Assembly Rings', 'Core Rings', 'Active Height',
             'Enrichment', 'Power MWt', 'Fuel Lifetime']].copy()
    df.columns = ['NA', 'NC', 'H', 'E', 'P', 'LT']
    df['LT'] = df['LT'].fillna(0)

    # Pre-compute physics factors and normalised lifetime L*
    df['F_A']   = df['NA'].apply(_assembly_factor)
    df['F_C']   = df['NC'].apply(_core_factor)
    df['L_star'] = df['LT'] * df['P'] / (df['F_A'] * df['F_C'] * df['H'])

    _train_df = df
    _feat_min = np.array([df['E'].min(),  df['NA'].min(), df['NC'].min(),
                          df['H'].min(),  df['P'].min()],  dtype=float)
    _feat_max = np.array([df['E'].max(),  df['NA'].max(), df['NC'].max(),
                          df['H'].max(),  df['P'].max()],  dtype=float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_gcmr_fuel_lifetime(assembly_rings, core_rings, active_height,
                                enrichment, power_mwt):
    """
    Estimate GCMR fuel lifetime (days) for a given design point.

    Parameters
    ----------
    assembly_rings : int or float
        Number of rings per fuel assembly N_A (training range: 4–7).
    core_rings : int or float
        Number of assembly rings in the core N_C (training range: 3–5).
    active_height : float
        Active core height in cm (training range: 40–385 cm).
    enrichment : float
        U-235 enrichment fraction, e.g. 0.12 for 12 %
        (training range: 0.05–0.1975).
    power_mwt : float
        Thermal power in MWt (training range: 1–60 MWt).

    Returns
    -------
    int
        Estimated fuel lifetime in days.
        Returns 0 if the model predicts subcritical conditions.
    """
    _load()

    df = _train_df

    # Normalise features to [0, 1] using training-data range
    feat_range = np.where(_feat_max != _feat_min, _feat_max - _feat_min, 1.0)
    q_norm = (np.array([enrichment, assembly_rings, core_rings,
                        active_height, power_mwt], dtype=float)
              - _feat_min) / feat_range

    train_norm = (df[['E', 'NA', 'NC', 'H', 'P']].values.astype(float)
                  - _feat_min) / feat_range

    # Find K nearest neighbours by Euclidean distance
    dists  = np.sqrt(((train_norm - q_norm) ** 2).sum(axis=1))
    k_idx  = np.argsort(dists)[:_K]
    nb     = df.iloc[k_idx]

    if len(nb) < 2:
        return 0

    # Fit L* = A1 * E + c by ordinary least squares on the K neighbours
    E_nb    = nb['E'].values
    Ls_nb   = nb['L_star'].values
    E_mean  = E_nb.mean()
    Ls_mean = Ls_nb.mean()
    denom   = ((E_nb - E_mean) ** 2).sum()

    if denom == 0:
        L_star_pred = max(0.0, Ls_mean)
    else:
        slope     = ((E_nb - E_mean) * (Ls_nb - Ls_mean)).sum() / denom
        intercept = Ls_mean - slope * E_mean

        if slope <= 0:
            return 0

        A2          = -intercept / slope
        L_star_pred = max(0.0, slope * (enrichment - A2))

    fa       = _assembly_factor(assembly_rings)
    fc       = _core_factor(core_rings)
    lifetime = L_star_pred * fa * fc * active_height / power_mwt

    return int(round(lifetime))


def estimate_gcmr_fuel_lifetime_from_params(params):
    """
    Convenience wrapper that reads inputs from a MOUSE params dict and
    writes the result back as params['Fuel Lifetime'].

    Required keys
    -------------
    'Assembly Rings' (or 'Number of Rings per Assembly'),
    'Core Rings', 'Active Height', 'Enrichment', 'Power MWt'
    """
    lifetime = estimate_gcmr_fuel_lifetime(
        assembly_rings = params.get('Assembly Rings',
                                    params.get('Number of Rings per Assembly')),
        core_rings     = params['Core Rings'],
        active_height  = params['Active Height'],
        enrichment     = params['Enrichment'],
        power_mwt      = params['Power MWt'],
    )
    params['Fuel Lifetime'] = lifetime
    return lifetime
