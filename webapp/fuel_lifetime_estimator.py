# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
LTMR fuel-lifetime estimator — local KNN regression.

Model
-----
    Lifetime = (3N²−3N+1) × H × A1 × (Enrichment − A2) / Power

For each query the K=4 nearest training points are found in the
(Enrichment, N, H, Power) feature space (raw features normalised to
[0,1] for the distance calculation).  A1 and A2 are then fitted by
simple linear regression of the normalised lifetime

    L* = Lifetime × Power / ((3N²−3N+1) × H)

against Enrichment on those K neighbours.  Subcritical cases have
L*=0 and are included in the training pool — they anchor the local
fit near the criticality boundary and improve A2 estimation.

Key detail
----------
N=6 neighbours are excluded when predicting N≥10, because the
small-assembly near-critical regime is physically discontinuous from
the rest of the design space.

Accuracy (5-fold CV on 247 non-zero training points, subcritical rows anchor boundary)
-----------------------------------------------------
  Median absolute error : ~110 days
  Median MAPE           : ~2.6 %   (N≥10 only: ~2–3 %)

Reference data
--------------
  assets/Ref_Results/LTMR_parametric_size_power_enrichment.xlsx
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data path and module-level cache
# ---------------------------------------------------------------------------
_XLSX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'assets', 'Ref_Results', 'LTMR_parametric_size_power_enrichment.xlsx',
)

_K = 4          # nearest neighbours to use for local fit

_train_df  = None   # cached training data (loaded once)
_feat_min  = None   # min of each feature — used for [0,1] normalisation
_feat_max  = None   # max of each feature


def _load():
    """Load and cache the training data on first call.

    All rows including subcritical (LT=0) are loaded — they anchor the
    local linear fit near the criticality boundary.
    """
    global _train_df, _feat_min, _feat_max
    if _train_df is not None:
        return

    df = pd.read_excel(_XLSX_PATH, sheet_name='Merged Data', engine='openpyxl')
    df = df[['Number of Rings per Assembly', 'Active Height',
             'Enrichment', 'Power MWt', 'Fuel Lifetime']].copy()
    df.columns = ['N', 'H', 'E', 'P', 'LT']
    df['LT'] = df['LT'].fillna(0)

    # Pre-compute N_pins and normalised lifetime L*
    # Subcritical cases (LT=0) get L*=0, anchoring the local fit.
    df['N_pins'] = 3 * df['N'] ** 2 - 3 * df['N'] + 1
    df['L_star'] = df['LT'] * df['P'] / (df['N_pins'] * df['H'])

    _train_df = df
    _feat_min = np.array([df['E'].min(), df['N'].min(),
                          df['H'].min(), df['P'].min()], dtype=float)
    _feat_max = np.array([df['E'].max(), df['N'].max(),
                          df['H'].max(), df['P'].max()], dtype=float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_ltmr_fuel_lifetime(n_rings_per_assembly, active_height,
                                enrichment, power_mwt):
    """
    Estimate LTMR fuel lifetime (days) for a given design point.

    Parameters
    ----------
    n_rings_per_assembly : int or float
        Number of rings per fuel assembly (training range: 6–24).
    active_height : float
        Active core height in cm (training range: 50–180 cm).
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

    # Exclude N=6 neighbours when predicting N≥10 — the small-assembly
    # near-critical regime is physically discontinuous from larger cores.
    if n_rings_per_assembly >= 10:
        df = df[df['N'] != 6]

    # Normalise features to [0, 1] using training-data range
    feat_range = np.where(_feat_max != _feat_min, _feat_max - _feat_min, 1.0)
    q_norm = (np.array([enrichment, n_rings_per_assembly,
                        active_height, power_mwt], dtype=float)
              - _feat_min) / feat_range

    train_norm = (df[['E', 'N', 'H', 'P']].values.astype(float)
                  - _feat_min) / feat_range

    # Find K nearest neighbours by Euclidean distance
    dists = np.sqrt(((train_norm - q_norm) ** 2).sum(axis=1))
    k_idx = np.argsort(dists)[:_K]
    nb = df.iloc[k_idx]

    if len(nb) < 2:
        return 0

    # Fit L* = A1 * E + c by ordinary least squares on the K neighbours
    E_nb     = nb['E'].values
    Ls_nb    = nb['L_star'].values
    E_mean   = E_nb.mean()
    Ls_mean  = Ls_nb.mean()
    denom    = ((E_nb - E_mean) ** 2).sum()

    if denom == 0:
        # All neighbours have identical enrichment — use their mean L*
        L_star_pred = max(0.0, Ls_mean)
    else:
        slope     = ((E_nb - E_mean) * (Ls_nb - Ls_mean)).sum() / denom
        intercept = Ls_mean - slope * E_mean

        if slope <= 0:
            return 0     # model implies no criticality at this enrichment

        A2          = -intercept / slope   # estimated critical enrichment
        L_star_pred = max(0.0, slope * (enrichment - A2))

    n_pins   = 3 * int(n_rings_per_assembly) ** 2 - 3 * int(n_rings_per_assembly) + 1
    lifetime = L_star_pred * n_pins * active_height / power_mwt

    return int(round(lifetime))


def estimate_ltmr_fuel_lifetime_from_params(params):
    """
    Convenience wrapper that reads inputs from a MOUSE params dict and
    writes the result back as params['Fuel Lifetime'].

    Required keys
    -------------
    'Number of Rings per Assembly', 'Active Height',
    'Enrichment', 'Power MWt'
    """
    lifetime = estimate_ltmr_fuel_lifetime(
        n_rings_per_assembly = params['Number of Rings per Assembly'],
        active_height        = params['Active Height'],
        enrichment           = params['Enrichment'],
        power_mwt            = params['Power MWt'],
    )
    params['Fuel Lifetime'] = lifetime
    return lifetime
