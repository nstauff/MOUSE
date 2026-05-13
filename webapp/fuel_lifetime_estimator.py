# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
LTMR fuel-lifetime estimator local KNN regression.

Model
-----
    Lifetime = (3N²−3N+1) × H × A1 × (Enrichment − A2) / Power

For each query the K=4 nearest training points are found in the
(Enrichment, N, H, Power) feature space (raw features normalised to
[0,1] for the distance calculation). A1 and A2 are then fitted by
simple linear regression of the normalised lifetime

    L* = Lifetime × Power / ((3N²−3N+1) × H)

against Enrichment on those K neighbours. Subcritical cases have
L*=0 and are included in the training pool they anchor the local
fit near the criticality boundary and improve A2 estimation.

Key detail
----------
N=6 neighbours are excluded when predicting N≥10, because the
small-assembly near-critical regime is physically discontinuous from
the rest of the design space.

Accuracy (5-fold CV on 247 non-zero training points, subcritical rows anchor boundary)
-----------------------------------------------------
  Median absolute error : ~110 days
  Median MAPE : ~2.6 % (N≥10 only: ~2-3 %)

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

_K = 4 # nearest neighbours to use for local fit

# Skip legacy precomputed cost columns from older parametric-study
# snapshots not read by the webapp (the cost engine is run fresh per
# request). Roughly halves the read time and lowers peak memory during
# the openpyxl parse.
_LEGACY_COST_PREFIXES = (
    'OCC_', 'OCC per kW_', 'OCC excl. fuel_', 'OCC excl. fuel per kW_',
    'TCI_', 'TCI per kW_',
    'AC_', 'AC per MWh_',
    'LCOE_',
)

def _keep_parametric_col(name):
    return not any(name.startswith(p) for p in _LEGACY_COST_PREFIXES)


_train_df = None # cached training data (loaded once)
_feat_min = None # min of each feature used for [0,1] normalisation
_feat_max = None # max of each feature


def _load():
    """Load and cache the training data on first call.

    All rows including subcritical (LT=0) are loaded they anchor the
    local linear fit near the criticality boundary.
    """
    global _train_df, _feat_min, _feat_max
    if _train_df is not None:
        return

    df = pd.read_excel(_XLSX_PATH, sheet_name='Merged Data', engine='openpyxl', usecols=_keep_parametric_col)
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
        Number of rings per fuel assembly (training range: 6-24).
    active_height : float
        Active core height in cm (training range: 50-180 cm).
    enrichment : float
        U-235 enrichment fraction, e.g. 0.12 for 12 %
        (training range: 0.05-0.1975).
    power_mwt : float
        Thermal power in MWt (training range: 1-60 MWt).

    Returns
    -------
    int
        Estimated fuel lifetime in days.
        Returns 0 if the model predicts subcritical conditions.
    """
    _load()

    df = _train_df

    # Exclude N=6 neighbours when predicting N≥10 the small-assembly
    # near-critical regime is physically discontinuous from larger cores.
    if n_rings_per_assembly >= 10:
        df = df[df['N'] != 6]

    # Normalise features to [0, 1] using training-data range
    feat_range = np.where(_feat_max != _feat_min, _feat_max - _feat_min, 1.0)
    q_full = np.array([enrichment, n_rings_per_assembly,
                       active_height, power_mwt], dtype=float)
    train_full = df[['E', 'N', 'H', 'P']].values.astype(float)

    # Power-independent subcritical pre-check. k_eff doesn't depend on
    # operating power, so dropping P (column index 3) from the distance
    # metric makes subcriticality consistent across P. Otherwise queries
    # identical in (N, H, E) but different in P could disagree on the
    # verdict.
    P_IDX = 3
    crit_idx = [i for i in range(len(_feat_min)) if i != P_IDX]
    crit_range = feat_range[crit_idx]
    q_norm_crit = (q_full[crit_idx] - _feat_min[crit_idx]) / crit_range
    train_norm_crit = ((train_full[:, crit_idx]
                        - _feat_min[crit_idx]) / crit_range)
    dists_crit = np.sqrt(((train_norm_crit - q_norm_crit) ** 2).sum(axis=1))
    k_idx_crit = np.argsort(dists_crit)[:_K]
    nb_crit = df.iloc[k_idx_crit]
    nb_dists_crit = dists_crit[k_idx_crit]
    if len(nb_crit) > 0:
        sub_mask_crit = (nb_crit['LT'].values <= 0)
        weights_crit = 1.0 / (nb_dists_crit + 1e-9)
        weights_crit = weights_crit / weights_crit.sum()
        # (a) Nearest-neighbour rule on geometry/E.
        if sub_mask_crit[0] and nb_dists_crit[0] < 0.1:
            return 0
        # (b) Majority-weight rule on geometry/E.
        if sub_mask_crit.any() and float(weights_crit[sub_mask_crit].sum()) >= 0.5:
            return 0

    q_norm = (q_full - _feat_min) / feat_range
    train_norm = (train_full - _feat_min) / feat_range

    # Find K nearest neighbours by Euclidean distance
    dists = np.sqrt(((train_norm - q_norm) ** 2).sum(axis=1))
    k_idx = np.argsort(dists)[:_K]
    nb = df.iloc[k_idx]

    if len(nb) < 2:
        return 0

    # Fit L* = A1 * E + c by ordinary least squares on the K neighbours
    E_nb = nb['E'].values
    Ls_nb = nb['L_star'].values
    E_mean = E_nb.mean()
    Ls_mean = Ls_nb.mean()
    denom = ((E_nb - E_mean) ** 2).sum()

    if denom == 0:
        # All neighbours have identical enrichment use their mean L*
        L_star_pred = max(0.0, Ls_mean)
    else:
        slope = ((E_nb - E_mean) * (Ls_nb - Ls_mean)).sum() / denom
        intercept = Ls_mean - slope * E_mean

        if slope <= 0:
            return 0 # model implies no criticality at this enrichment

        A2 = -intercept / slope # estimated critical enrichment
        L_star_pred = max(0.0, slope * (enrichment - A2))

    n_pins = 3 * int(n_rings_per_assembly) ** 2 - 3 * int(n_rings_per_assembly) + 1
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
        active_height = params['Active Height'],
        enrichment = params['Enrichment'],
        power_mwt = params['Power MWt'],
    )
    params['Fuel Lifetime'] = lifetime
    return lifetime


# ---------------------------------------------------------------------------
# Max Peaking Factor distance-weighted KNN interpolation
# ---------------------------------------------------------------------------

def _ltmr_knn_scalar(column_name, n_rings_per_assembly, active_height,
                     enrichment, power_mwt):
    """Generic distance-weighted KNN interpolator for any scalar LTMR
    parametric-study column. Uses the K=4 nearest neighbours in the same
    (E, N, H, P) feature space as the lifetime estimator. Rows with NaN
    in the requested column are skipped (so subcritical rows that don't
    carry e.g. a peaking factor don't pollute the average)."""
    _load()

    df = _train_df.copy()
    if column_name not in df.columns:
        full = pd.read_excel(_XLSX_PATH, sheet_name='Merged Data',
                             engine='openpyxl')
        df[column_name] = full[column_name].values
        _train_df[column_name] = df[column_name]

    df = df[df[column_name].notna()].reset_index(drop=True)
    if len(df) == 0:
        return 0.0

    if n_rings_per_assembly >= 10:
        df = df[df['N'] != 6].reset_index(drop=True)
    if len(df) == 0:
        return 0.0

    feat_range = np.where(_feat_max != _feat_min, _feat_max - _feat_min, 1.0)
    q_norm = (np.array([enrichment, n_rings_per_assembly,
                        active_height, power_mwt], dtype=float)
              - _feat_min) / feat_range
    train_norm = (df[['E', 'N', 'H', 'P']].values.astype(float)
                  - _feat_min) / feat_range
    dists = np.sqrt(((train_norm - q_norm) ** 2).sum(axis=1))
    k_idx = np.argsort(dists)[: _K]
    nb = df.iloc[k_idx]
    nb_dists = dists[k_idx]
    weights = 1.0 / (nb_dists + 1e-9)
    weights = weights / weights.sum()
    return float(np.sum(weights * nb[column_name].values))


def get_ltmr_peaking_factor(n_rings_per_assembly, active_height,
                            enrichment, power_mwt):
    """Interpolated Max Peaking Factor for an LTMR design point."""
    return _ltmr_knn_scalar('Max Peaking Factor',
                            n_rings_per_assembly, active_height,
                            enrichment, power_mwt)


def get_ltmr_axial_leakage_pct(n_rings_per_assembly, active_height,
                               enrichment, power_mwt):
    """Interpolated BOL axial leakage (%) for an LTMR design point.
    Geometric depends mostly on Active Height; weakly on the rest."""
    return _ltmr_knn_scalar('Estimated Axial Leakage (%)',
                            n_rings_per_assembly, active_height,
                            enrichment, power_mwt)


def get_ltmr_total_leakage_pct(n_rings_per_assembly, active_height,
                               enrichment, power_mwt):
    """Interpolated BOL total leakage (%) for an LTMR design point.
    Geometric depends on both Active Height and core radius (≈ N)."""
    return _ltmr_knn_scalar('Estimated Total Leakage (%)',
                            n_rings_per_assembly, active_height,
                            enrichment, power_mwt)


# ---------------------------------------------------------------------------
# Physics-based leakage with KNN/physics dispatch
# ---------------------------------------------------------------------------
# Migration area calibrated against one mid-range LTMR training case
# (N=18, H=110, total leakage 11.14 %): see derivation in commit history.
_LTMR_M_SQUARED_CM2 = 60.0
# Effective reflector savings fraction of reflector thickness that
# extends the diffusion problem ("reflector savings" δ). Calibrated so
# the physics formula reproduces the parametric-study leakage at trained
# boundary cases to within ~15 %.
_LTMR_REFLECTOR_SAVINGS = 0.55


def _ltmr_physics_leakage(active_radius_cm, active_height_cm,
                          radial_reflector_cm, axial_reflector_cm):
    """Physics-based estimate of (axial%, total%) leakage for LTMR using
    the one-group migration-area approximation:
        B² = (2.405/R_eff)² + (π/H_eff)²
        P_NL = 1 / (1 + M² × B²)
    Reflector savings δ are added to R and H (top and bottom for axial).
    """
    delta_r = _LTMR_REFLECTOR_SAVINGS * radial_reflector_cm
    delta_z = _LTMR_REFLECTOR_SAVINGS * axial_reflector_cm
    R_eff = active_radius_cm + delta_r
    H_eff = active_height_cm + 2.0 * delta_z

    B2_radial = (2.405 / R_eff) ** 2
    B2_axial = (np.pi / H_eff) ** 2
    B2_total = B2_radial + B2_axial

    P_NL = 1.0 / (1.0 + _LTMR_M_SQUARED_CM2 * B2_total)
    total_lk = (1.0 - P_NL) * 100.0
    axial_lk = total_lk * (B2_axial / B2_total) # split by buckling
    return axial_lk, total_lk


def _ltmr_h_within_trained_range(n_rings_per_assembly, active_height):
    """True if the user's H falls inside the per-N trained H range
    (slight 5 % tolerance on each side). For untrained N values the
    range is computed from the nearest trained N."""
    _load()
    df = _train_df.dropna(subset=['H'])
    df = df[df['LT'] > 0]
    if len(df) == 0:
        return False
    trained_ns = sorted(df['N'].unique())
    nearest_n = min(trained_ns, key=lambda x: abs(x - n_rings_per_assembly))
    h_vals = df[df['N'] == nearest_n]['H']
    if len(h_vals) == 0:
        return False
    h_min, h_max = h_vals.min(), h_vals.max()
    return (h_min * 0.95) <= active_height <= (h_max * 1.05)


def get_ltmr_leakage(n_rings_per_assembly, active_height, enrichment, power_mwt,
                     active_radius_cm, radial_reflector_cm, axial_reflector_cm):
    """
    Return (axial_pct, total_pct, source) for LTMR.

    source = 'interpolated' if the user's H is within the trained
             per-N range, otherwise 'physics' in which case the
             values come from the migration-area physics formula
             instead of the KNN average (which would saturate at
             the boundary).
    """
    if _ltmr_h_within_trained_range(n_rings_per_assembly, active_height):
        ax = get_ltmr_axial_leakage_pct(n_rings_per_assembly, active_height,
                                        enrichment, power_mwt)
        tot = get_ltmr_total_leakage_pct(n_rings_per_assembly, active_height,
                                         enrichment, power_mwt)
        return ax, tot, 'interpolated'

    ax, tot = _ltmr_physics_leakage(active_radius_cm, active_height,
                                    radial_reflector_cm, axial_reflector_cm)
    return ax, tot, 'physics'


# ---------------------------------------------------------------------------
# keff(time) curve loader and interpolator
# ---------------------------------------------------------------------------
_curve_df = None # cached training data with parsed depletion curves


def _parse_curve(raw):
    """Parse the string-formatted list stored in the Excel cell into a numpy
    array of floats. Returns None for missing / unparseable cells."""
    import ast
    if raw is None:
        return None
    if isinstance(raw, float) and np.isnan(raw):
        return None
    try:
        vals = ast.literal_eval(str(raw).strip())
    except (ValueError, SyntaxError):
        return None
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return arr


def _load_curves():
    """Lazy-load the depletion-curve dataframe (one row per training case
    that has a non-trivial keff(time) curve)."""
    global _curve_df
    if _curve_df is not None:
        return _curve_df

    df = pd.read_excel(_XLSX_PATH, sheet_name='Merged Data', engine='openpyxl', usecols=_keep_parametric_col)
    df = df[['Number of Rings per Assembly', 'Active Height',
             'Enrichment', 'Power MWt', 'Fuel Lifetime',
             'Depletion Time Steps', 'keff 3D (2D corrected)']].copy()
    df.columns = ['N', 'H', 'E', 'P', 'LT', 'time_raw', 'keff_raw']

    df['time'] = df['time_raw'].apply(_parse_curve)
    df['keff'] = df['keff_raw'].apply(_parse_curve)
    df = df.drop(columns=['time_raw', 'keff_raw'])

    # Keep only rows where both arrays are valid AND have matching lengths
    valid = df.apply(lambda r: r['time'] is not None
                               and r['keff'] is not None
                               and len(r['time']) == len(r['keff'])
                               and len(r['time']) >= 2,
                     axis=1)
    df = df[valid].reset_index(drop=True)

    _curve_df = df
    return _curve_df


def get_ltmr_keff_curve(n_rings_per_assembly, active_height,
                        enrichment, power_mwt,
                        anchor_lifetime_days=None):
    """
    Build an interpolated keff vs time curve for a given LTMR design point.

    For each timestep index i, time[i] and keff[i] are independently
    interpolated as a distance-weighted average over the K=4 nearest
    training neighbours (in normalised E, N, H, P space same metric as
    the lifetime estimator). Only training rows that actually carry a
    keff curve are used. The returned curve is truncated where keff
    drops to 1.0 (the user is not interested in subcritical tail).

    Parameters
    ----------
    anchor_lifetime_days : float, optional
        If provided, the time axis is linearly rescaled so the curve's
        keff = 1 crossing lands exactly on this value (in days). This
        keeps the keff curve consistent with the separately estimated
        fuel lifetime (the keff interpolation by itself often disagrees
        with the lifetime estimator by 20-40 %).

    Returns
    -------
    times_days : np.ndarray
    keffs : np.ndarray
        Aligned arrays. Empty arrays are returned if no usable
        training data is available (e.g. the query is so far from
        the design space that no neighbour has a curve).
    """
    df = _load_curves()
    if len(df) == 0:
        return np.array([]), np.array([])

    # Same neighbour rule as the lifetime estimator
    if n_rings_per_assembly >= 10:
        df = df[df['N'] != 6].reset_index(drop=True)

    if len(df) == 0:
        return np.array([]), np.array([])

    # Normalise features to [0, 1] using training-data range
    _load() # ensures _feat_min / _feat_max are populated from full dataset
    feat_range = np.where(_feat_max != _feat_min, _feat_max - _feat_min, 1.0)
    q_norm = (np.array([enrichment, n_rings_per_assembly,
                        active_height, power_mwt], dtype=float)
              - _feat_min) / feat_range
    train_norm = (df[['E', 'N', 'H', 'P']].values.astype(float)
                  - _feat_min) / feat_range
    dists = np.sqrt(((train_norm - q_norm) ** 2).sum(axis=1))
    k_idx = np.argsort(dists)[: _K]
    nb = df.iloc[k_idx]
    nb_dists = dists[k_idx]

    # Distance-weighted interpolation. Use 1/(d + tiny) weights so an
    # exact match (d=0) dominates without divide-by-zero.
    weights = 1.0 / (nb_dists + 1e-9)
    weights = weights / weights.sum()

    # Time-step axis is whatever length the neighbour curves have. They
    # all use the same template in the parametric study (18 steps),
    # but be defensive and use the minimum length across the chosen
    # neighbours.
    nsteps = min(len(t) for t in nb['time'].values)
    times = np.zeros(nsteps)
    keffs = np.zeros(nsteps)
    for w, t_arr, k_arr in zip(weights, nb['time'].values, nb['keff'].values):
        times += w * t_arr[:nsteps]
        keffs += w * k_arr[:nsteps]

    # Truncate at the first crossing of keff = 1 (going down). Linear-
    # interpolate the exact crossing time.
    times_out = [times[0]]
    keffs_out = [keffs[0]]
    for i in range(1, nsteps):
        t_prev, k_prev = times_out[-1], keffs_out[-1]
        t_cur, k_cur = times[i], keffs[i]
        if k_cur >= 1.0:
            times_out.append(t_cur)
            keffs_out.append(k_cur)
        else:
            # Linear-interpolate the crossing keff = 1
            if k_prev > k_cur:
                frac = (k_prev - 1.0) / (k_prev - k_cur)
                t_cross = t_prev + frac * (t_cur - t_prev)
                times_out.append(t_cross)
                keffs_out.append(1.0)
            break

    times_arr = np.asarray(times_out)
    keffs_arr = np.asarray(keffs_out)

    # Anchor the curve's k=1 crossing to the externally-estimated
    # fuel lifetime, if requested. We linearly scale the time axis so
    # the last point of times_arr (which is at k=1 by construction
    # whenever the curve crosses 1) becomes anchor_lifetime_days. This
    # preserves the curve's shape but makes it consistent with the
    # KNN lifetime estimator that the webapp shows in its info card.
    if (anchor_lifetime_days is not None
            and anchor_lifetime_days > 0
            and times_arr.size >= 2
            and keffs_arr[-1] <= 1.0 + 1e-9
            and times_arr[-1] > 0):
        scale = float(anchor_lifetime_days) / float(times_arr[-1])
        times_arr = times_arr * scale

    return times_arr, keffs_arr
