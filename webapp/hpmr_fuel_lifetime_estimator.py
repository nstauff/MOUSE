"""HPMR fuel-lifetime estimator using KNN local fit on a physics-constrained
normalized lifetime.

Mirrors the LTMR / GCMR estimators
(`fuel_lifetime_estimator.py`, `gcmr_fuel_lifetime_estimator.py`) but
uses HPMR's specific geometry-to-fuel-mass relationship.

Physics scaling
---------------
For HPMR, the total fuel-pin count and active height are determined
by `Assembly Rings` (N_A), `Core Rings` (N_C), and the active height H:

    N_pins(N_A, N_C) = (3·N_C² − 3·N_C) × f_HPMR(N_A)
    f_HPMR(N_A)     = total_positions(N_A) − total_positions(ceil(N_A/2))

`total_positions(r)` is the standard hex-grid count for `r` rings per
edge (matches `core_design.utils.calculate_number_of_rings`).

Total uranium scales as `M_U = U_per_pin_cm × N_pins × H`.  Calibrated
from the parametric study at 1.6116 g/(pin·cm) — this constant is
extraordinarily stable across the 84 training rows (CV < 0.04 %).

The physics-constrained normalised lifetime is

    L* = LT × P / (N_pins(N_A, N_C) × H × E)

L* is approximately constant across the design space; residual
variation is captured by the K = 4 distance-weighted KNN.

Training data
-------------
`assets/Ref_Results/HPMR_parametric_size_power_enrichment.xlsx`,
sheet `Sheet1`.  84 rows covering:
  * Assembly Rings: 6 (single value — NA-extrapolation relies on the
    physics scaling alone)
  * Core Rings:    [3, 4, 5, 6, 7]
  * Height:        15 unique values, 136 – 1056 cm
  * Enrichment:    [0.10, 0.13, 0.16, 0.1975]
  * Power MWt:     [1, 5, 20, 60]
  * Fuel Lifetime: 0 to 11,158 days (4 subcritical rows)
"""
import os
import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_XLSX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'assets', 'Ref_Results', 'HPMR_parametric_size_power_enrichment.xlsx',
)
_SHEET = 'Sheet1'

# Total uranium per (fuel pin · cm).  Derived from the parametric
# study: M_U / (N_pins × H) is 1.6116 ± 0.0006 across all 84 rows.
HPMR_U_PER_PIN_CM = 1.6116   # g of total uranium per (fuel pin · cm)

_K = 4
_train_df = None
_feat_min = None
_feat_max = None


# ---------------------------------------------------------------------------
# HPMR geometry helpers (replicated locally to avoid the openmc import
# chain that core_design.utils pulls in)
# ---------------------------------------------------------------------------

def _hex_positions(r):
    """Total cell positions in an r-ring hex assembly.  Matches
    core_design.utils.calculate_number_of_rings."""
    r = int(r)
    return 2 * r * (r - 1) + 2 * sum(range(1, r - 1)) + 2 * r - 1


def _fuel_pins_per_assembly(n_assembly_rings):
    """Matches core_design.utils.calculate_number_fuel_elements_hpmr."""
    n = int(n_assembly_rings)
    return _hex_positions(n) - _hex_positions(int(math.ceil(n / 2)))


def hpmr_total_fuel_pins(n_assembly_rings, n_core_rings):
    """Total fuel-pin count for HPMR at (N_A, N_C)."""
    n_assemblies = (3 * int(n_core_rings) ** 2) - (3 * int(n_core_rings))
    return _fuel_pins_per_assembly(n_assembly_rings) * n_assemblies


def hpmr_total_uranium_mass_g(n_assembly_rings, n_core_rings, active_height_cm):
    """Total uranium mass (g) for an HPMR core at the given geometry."""
    return HPMR_U_PER_PIN_CM * hpmr_total_fuel_pins(
        n_assembly_rings, n_core_rings,
    ) * float(active_height_cm)


# ---------------------------------------------------------------------------
# Training data load
# ---------------------------------------------------------------------------

def _load():
    """Lazy-load HPMR rows from the parametric-study XLSX and pre-compute
    the physics factors and normalised lifetime."""
    global _train_df, _feat_min, _feat_max
    if _train_df is not None:
        return

    df = pd.read_excel(_XLSX_PATH, sheet_name=_SHEET, engine='openpyxl')
    # Map the workbook column names to the short names used internally
    df = df[['Assembly Rings', 'Core Rings', 'Height',
             'Enrichment', 'Power MWt', 'Fuel Lifetime']].copy()
    df.columns = ['NA', 'NC', 'H', 'E', 'P', 'LT']
    df['LT'] = df['LT'].fillna(0)

    # Physics factor and normalised lifetime
    df['N_pins'] = df.apply(lambda r: hpmr_total_fuel_pins(int(r['NA']),
                                                          int(r['NC'])),
                            axis=1)
    df['F_geom'] = df['N_pins'].astype(float) * df['H'].astype(float)
    df['L_star'] = np.where(
        (df['LT'] > 0) & (df['E'] > 0),
        df['LT'] * df['P'] / (df['F_geom'] * df['E']),
        0.0,
    )

    _train_df = df
    _feat_min = np.array([df['E'].min(),  df['NA'].min(), df['NC'].min(),
                          df['H'].min(),  df['P'].min()],  dtype=float)
    _feat_max = np.array([df['E'].max(),  df['NA'].max(), df['NC'].max(),
                          df['H'].max(),  df['P'].max()],  dtype=float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_hpmr_fuel_lifetime(n_rings_per_assembly, n_rings_per_core,
                                active_height, enrichment, power_mwt):
    """KNN local fit on the HPMR parametric study.

    Parameters
    ----------
    n_rings_per_assembly : int  (N_A — number of fuel-pin rings per assembly)
    n_rings_per_core     : int  (N_C — number of assembly rings per core)
    active_height        : float (cm)
    enrichment           : float  (fraction, 0–1)
    power_mwt            : float  (thermal power, MWt)

    Returns
    -------
    fuel_lifetime_days : int
        Estimated days of full-power operation before fuel depletion.
        Clamped to ≥ 0; returns 0 for subcritical / sub-threshold cases.
    """
    _load()
    df = _train_df
    if df is None or len(df) == 0:
        return 0

    # KNN distance only over features that vary in the training data.
    # In the current dataset N_A is constant (= 6), so it would
    # contribute nothing to neighbour selection.  N_C, H, E, P all
    # vary and feed into the distance.  The physics scaling
    # (L* × N_pins · H · E / P) handles N_A extrapolation by itself.
    feat_idx_active = [i for i in range(5) if _feat_max[i] > _feat_min[i]]
    if not feat_idx_active:
        return 0
    feat_range = (_feat_max - _feat_min)
    feat_range_active = feat_range[feat_idx_active]

    q_full = np.array([float(enrichment),
                       float(n_rings_per_assembly),
                       float(n_rings_per_core),
                       float(active_height),
                       float(power_mwt)], dtype=float)
    q_norm = (q_full[feat_idx_active]
              - _feat_min[feat_idx_active]) / feat_range_active
    train_full = df[['E', 'NA', 'NC', 'H', 'P']].values.astype(float)
    train_norm = ((train_full[:, feat_idx_active]
                   - _feat_min[feat_idx_active]) / feat_range_active)
    dists = np.sqrt(((train_norm - q_norm) ** 2).sum(axis=1))

    k_idx = np.argsort(dists)[: _K]
    nb = df.iloc[k_idx]
    nb_dists = dists[k_idx]

    # Distance-weighted neighbour averaging
    weights = 1.0 / (nb_dists + 1e-9)
    weights /= weights.sum()

    # Subcritical detection — two-pronged:
    #   (a) Nearest-neighbour rule: if the closest training point is
    #       subcritical (LT=0) and the query is essentially right on
    #       top of it (normalised distance < 0.1), declare subcritical.
    #       Catches the sharp criticality cliff cleanly.
    #   (b) Majority weight rule: if the distance-weighted share of
    #       subcritical neighbours is ≥ 50 %, declare subcritical.
    sub_mask = (nb['LT'].values <= 0)
    if sub_mask[0] and nb_dists[0] < 0.1:
        return 0
    sub_weight_fraction = (float(weights[sub_mask].sum())
                           if sub_mask.any() else 0.0)
    if sub_weight_fraction >= 0.5:
        return 0

    l_star_weighted = float((weights * nb['L_star'].values).sum())
    if l_star_weighted <= 0.0:
        return 0

    # Reconstruct LT from L*
    n_pins_q = hpmr_total_fuel_pins(n_rings_per_assembly, n_rings_per_core)
    f_geom_q = float(n_pins_q) * float(active_height)
    if f_geom_q <= 0 or enrichment <= 0:
        return 0
    lt_predicted = (l_star_weighted * f_geom_q
                    * float(enrichment) / float(power_mwt))
    if lt_predicted <= 0:
        return 0
    return int(round(lt_predicted))


def populate_hpmr_lifetime(params):
    """Convenience wrapper used by the webapp builder.  Reads
    geometry / E / P from params and writes 'Fuel Lifetime'."""
    lifetime = estimate_hpmr_fuel_lifetime(
        n_rings_per_assembly = params.get('Number of Rings per Assembly',
                                          params.get('Assembly Rings')),
        n_rings_per_core     = params.get('Number of Rings per Core',
                                          params.get('Core Rings')),
        active_height        = params['Active Height'],
        enrichment           = params['Enrichment'],
        power_mwt            = params['Power MWt'],
    )
    params['Fuel Lifetime'] = lifetime
    return lifetime
