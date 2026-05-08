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
    f_HPMR(N_A) = total_positions(N_A) − total_positions(ceil(N_A/2))

`total_positions(r)` is the standard hex-grid count for `r` rings per
edge (matches `core_design.utils.calculate_number_of_rings`).

Total uranium scales as `M_U = U_per_pin_cm × N_pins × H`. Calibrated
from the parametric study at 1.6116 g/(pin·cm) this constant is
extraordinarily stable across the 84 training rows (CV < 0.04 %).

The physics-constrained normalised lifetime is

    L* = LT × P / (N_pins(N_A, N_C) × H × E)

L* is approximately constant across the design space; residual
variation is captured by the K = 4 distance-weighted KNN.

Training data
-------------
`assets/Ref_Results/HPMR_parametric_size_power_enrichment.xlsx`,
sheet `Sheet1`. 84 rows covering:
  * Assembly Rings: 6 (single value NA-extrapolation relies on the
    physics scaling alone)
  * Core Rings: [3, 4, 5, 6, 7]
  * Height: 15 unique values, 136 1056 cm
  * Enrichment: [0.10, 0.13, 0.16, 0.1975]
  * Power MWt: [1, 5, 20, 60]
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

# Total uranium per (fuel pin · cm). Derived from the parametric
# study: M_U / (N_pins × H) is 1.6116 ± 0.0006 across all 84 rows.
HPMR_U_PER_PIN_CM = 1.6116 # g of total uranium per (fuel pin · cm)

_K = 4
_train_df = None
_feat_min = None
_feat_max = None


# ---------------------------------------------------------------------------
# HPMR geometry helpers (replicated locally to avoid the openmc import
# chain that core_design.utils pulls in)
# ---------------------------------------------------------------------------

def _hex_positions(r):
    """Total cell positions in an r-ring hex assembly. Matches
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
    _feat_min = np.array([df['E'].min(), df['NA'].min(), df['NC'].min(),
                          df['H'].min(), df['P'].min()], dtype=float)
    _feat_max = np.array([df['E'].max(), df['NA'].max(), df['NC'].max(),
                          df['H'].max(), df['P'].max()], dtype=float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_hpmr_fuel_lifetime(n_rings_per_assembly, n_rings_per_core,
                                active_height, enrichment, power_mwt):
    """KNN local fit on the HPMR parametric study.

    Parameters
    ----------
    n_rings_per_assembly : int (N_A number of fuel-pin rings per assembly)
    n_rings_per_core : int (N_C number of assembly rings per core)
    active_height : float (cm)
    enrichment : float (fraction, 0-1)
    power_mwt : float (thermal power, MWt)

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
    # contribute nothing to neighbour selection. N_C, H, E, P all
    # vary and feed into the distance. The physics scaling
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
    train_full = df[['E', 'NA', 'NC', 'H', 'P']].values.astype(float)

    # Power-independent subcritical pre-check. k_eff depends on
    # geometry/E only — not on operating power — so subcriticality must
    # not depend on P. Including P in the neighbour metric caused two
    # queries identical in (NA, NC, H, E) but different in P to disagree
    # on the verdict (P=1 → 0 days, P=60 → small positive days). Drop P
    # (column index 4) from the distance metric for this check.
    P_IDX = 4
    crit_idx = [i for i in feat_idx_active if i != P_IDX]
    if crit_idx:
        crit_range = feat_range[crit_idx]
        q_norm_crit = (q_full[crit_idx] - _feat_min[crit_idx]) / crit_range
        train_norm_crit = ((train_full[:, crit_idx]
                            - _feat_min[crit_idx]) / crit_range)
        dists_crit = np.sqrt(((train_norm_crit - q_norm_crit) ** 2).sum(axis=1))
        k_idx_crit = np.argsort(dists_crit)[:_K]
        nb_crit = df.iloc[k_idx_crit]
        nb_dists_crit = dists_crit[k_idx_crit]
        sub_mask_crit = (nb_crit['LT'].values <= 0)
        weights_crit = 1.0 / (nb_dists_crit + 1e-9)
        weights_crit = weights_crit / weights_crit.sum()
        # (a) Nearest-neighbour rule on geometry/E.
        if sub_mask_crit[0] and nb_dists_crit[0] < 0.1:
            return 0
        # (b) Majority-weight rule on geometry/E.
        if sub_mask_crit.any() and float(weights_crit[sub_mask_crit].sum()) >= 0.5:
            return 0

    q_norm = (q_full[feat_idx_active]
              - _feat_min[feat_idx_active]) / feat_range_active
    train_norm = ((train_full[:, feat_idx_active]
                   - _feat_min[feat_idx_active]) / feat_range_active)
    dists = np.sqrt(((train_norm - q_norm) ** 2).sum(axis=1))

    k_idx = np.argsort(dists)[: _K]
    nb = df.iloc[k_idx]
    nb_dists = dists[k_idx]

    # Distance-weighted neighbour averaging
    weights = 1.0 / (nb_dists + 1e-9)
    weights /= weights.sum()

    # Subcritical detection two-pronged:
    # (a) Nearest-neighbour rule: if the closest training point is
    # subcritical (LT=0) and the query is essentially right on
    # top of it (normalised distance < 0.1), declare subcritical.
    # Catches the sharp criticality cliff cleanly.
    # (b) Majority weight rule: if the distance-weighted share of
    # subcritical neighbours is ≥ 50 %, declare subcritical.
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
    """Convenience wrapper used by the webapp builder. Reads
    geometry / E / P from params and writes 'Fuel Lifetime'."""
    lifetime = estimate_hpmr_fuel_lifetime(
        n_rings_per_assembly = params.get('Number of Rings per Assembly',
                                          params.get('Assembly Rings')),
        n_rings_per_core = params.get('Number of Rings per Core',
                                          params.get('Core Rings')),
        active_height = params['Active Height'],
        enrichment = params['Enrichment'],
        power_mwt = params['Power MWt'],
    )
    params['Fuel Lifetime'] = lifetime
    return lifetime


# ---------------------------------------------------------------------------
# k_eff(time) curve same interpolation pattern as the LTMR / GCMR versions
# ---------------------------------------------------------------------------

_curve_df = None # cached subset of training rows that carry a k_eff curve


def _parse_curve(raw):
    """Parse a string-formatted list (e.g. '[1.0, 0.99, 0.97, ...]') from a
    cell into a numpy array of floats. Returns None for missing / bad cells."""
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
    that has a non-trivial k_eff(time) curve)."""
    global _curve_df
    if _curve_df is not None:
        return _curve_df

    df = pd.read_excel(_XLSX_PATH, sheet_name=_SHEET, engine='openpyxl')
    df = df[['Assembly Rings', 'Core Rings', 'Height',
             'Enrichment', 'Power MWt', 'Fuel Lifetime',
             'Depletion Time Steps', 'keff 3D (2D corrected)']].copy()
    df.columns = ['NA', 'NC', 'H', 'E', 'P', 'LT', 'time_raw', 'keff_raw']

    df['time'] = df['time_raw'].apply(_parse_curve)
    df['keff'] = df['keff_raw'].apply(_parse_curve)
    df = df.drop(columns=['time_raw', 'keff_raw'])

    valid = df.apply(lambda r: r['time'] is not None
                               and r['keff'] is not None
                               and len(r['time']) == len(r['keff'])
                               and len(r['time']) >= 2,
                     axis=1)
    df = df[valid].reset_index(drop=True)

    _curve_df = df
    return _curve_df


def get_hpmr_keff_curve(n_rings_per_assembly, n_rings_per_core, active_height,
                        enrichment, power_mwt,
                        anchor_lifetime_days=None):
    """Build an interpolated k_eff vs time curve for a given HPMR design point.

    Mirrors the LTMR / GCMR versions: distance-weighted average of the K=4
    nearest training neighbours in normalised feature space. KNN distance
    only uses features that vary in the training data N_A is constant
    (=6) in the current parametric study, so it's excluded from the metric
    and only re-enters via the physics scaling on the lifetime axis.

    The curve is truncated at the first crossing of k_eff = 1. Optional
    `anchor_lifetime_days` rescales the time axis so that crossing matches
    the externally-estimated fuel lifetime.

    Returns
    -------
    times_days : np.ndarray
    keffs : np.ndarray
        Empty arrays if no usable training data is available.
    """
    df = _load_curves()
    if len(df) == 0:
        return np.array([]), np.array([])

    _load() # populate _feat_min / _feat_max
    feat_idx_active = [i for i in range(5) if _feat_max[i] > _feat_min[i]]
    if not feat_idx_active:
        return np.array([]), np.array([])
    feat_range_active = (_feat_max - _feat_min)[feat_idx_active]

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

    weights = 1.0 / (nb_dists + 1e-9)
    weights = weights / weights.sum()

    nsteps = min(len(t) for t in nb['time'].values)
    times = np.zeros(nsteps)
    keffs = np.zeros(nsteps)
    for w, t_arr, k_arr in zip(weights, nb['time'].values, nb['keff'].values):
        times += w * t_arr[:nsteps]
        keffs += w * k_arr[:nsteps]

    # Truncate at k_eff = 1 crossing (linear-interpolate exact crossing)
    times_out = [times[0]]
    keffs_out = [keffs[0]]
    for i in range(1, nsteps):
        t_prev, k_prev = times_out[-1], keffs_out[-1]
        t_cur, k_cur = times[i], keffs[i]
        if k_cur >= 1.0:
            times_out.append(t_cur)
            keffs_out.append(k_cur)
        else:
            if k_prev > k_cur:
                frac = (k_prev - 1.0) / (k_prev - k_cur)
                t_cross = t_prev + frac * (t_cur - t_prev)
                times_out.append(t_cross)
                keffs_out.append(1.0)
            break

    times_arr = np.asarray(times_out)
    keffs_arr = np.asarray(keffs_out)

    if (anchor_lifetime_days is not None
            and anchor_lifetime_days > 0
            and times_arr.size >= 2
            and keffs_arr[-1] <= 1.0 + 1e-9
            and times_arr[-1] > 0):
        scale = float(anchor_lifetime_days) / float(times_arr[-1])
        times_arr = times_arr * scale

    return times_arr, keffs_arr


# ---------------------------------------------------------------------------
# Generic KNN interpolator for any scalar HPMR column
# ---------------------------------------------------------------------------

def _hpmr_knn_scalar(column_name, n_rings_per_assembly, n_rings_per_core,
                     active_height, enrichment, power_mwt):
    """Distance-weighted KNN interpolator for any scalar HPMR
    parametric-study column. Uses the K=4 nearest neighbours in the
    same active feature subset as the lifetime estimator (drops
    degenerate-range features so off-grid geometry queries don't get
    swamped by a constant offset)."""
    _load()

    df = _train_df.copy()
    if column_name not in df.columns:
        full = pd.read_excel(_XLSX_PATH, sheet_name=_SHEET, engine='openpyxl')
        df[column_name] = full[column_name].values
        _train_df[column_name] = df[column_name]

    df = df[df[column_name].notna()].reset_index(drop=True)
    if len(df) == 0:
        return 0.0

    feat_idx_active = [i for i in range(5) if _feat_max[i] > _feat_min[i]]
    if not feat_idx_active:
        return 0.0
    feat_range_active = (_feat_max - _feat_min)[feat_idx_active]

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
    weights = 1.0 / (nb_dists + 1e-9)
    weights = weights / weights.sum()
    return float(np.sum(weights * nb[column_name].values))


def get_hpmr_peaking_factor(n_rings_per_assembly, n_rings_per_core,
                            active_height, enrichment, power_mwt):
    """Interpolated Max Peaking Factor for an HPMR design point."""
    return _hpmr_knn_scalar('Max Peaking Factor',
                            n_rings_per_assembly, n_rings_per_core,
                            active_height, enrichment, power_mwt)


def get_hpmr_axial_leakage_pct(n_rings_per_assembly, n_rings_per_core,
                               active_height, enrichment, power_mwt):
    """Interpolated BOL axial leakage (%) for an HPMR design point."""
    return _hpmr_knn_scalar('Estimated Axial Leakage (%)',
                            n_rings_per_assembly, n_rings_per_core,
                            active_height, enrichment, power_mwt)


def get_hpmr_total_leakage_pct(n_rings_per_assembly, n_rings_per_core,
                               active_height, enrichment, power_mwt):
    """Interpolated BOL total leakage (%) for an HPMR design point."""
    return _hpmr_knn_scalar('Estimated Total Leakage (%)',
                            n_rings_per_assembly, n_rings_per_core,
                            active_height, enrichment, power_mwt)


# ---------------------------------------------------------------------------
# Physics-based leakage with KNN/physics dispatch
# ---------------------------------------------------------------------------
# HPMR is graphite-moderated like GCMR (monolith graphite block),
# so the migration area and reflector savings are similar to GCMR's
# values. M² calibrated against the typical (NA=6, NC=5) HPMR
# parametric case.
_HPMR_M_SQUARED_CM2 = 220.0
_HPMR_REFLECTOR_SAVINGS = 0.65


def _hpmr_physics_leakage(active_radius_cm, active_height_cm,
                          radial_reflector_cm, axial_reflector_cm):
    """Physics-based (axial%, total%) leakage using the one-group
    migration-area approximation. Same formula as the GCMR fallback
    with HPMR-calibrated constants."""
    delta_r = _HPMR_REFLECTOR_SAVINGS * radial_reflector_cm
    delta_z = _HPMR_REFLECTOR_SAVINGS * axial_reflector_cm
    R_eff = active_radius_cm + delta_r
    H_eff = active_height_cm + 2.0 * delta_z
    B2_radial = (2.405 / R_eff) ** 2
    B2_axial = (np.pi / H_eff) ** 2
    B2_total = B2_radial + B2_axial
    P_NL = 1.0 / (1.0 + _HPMR_M_SQUARED_CM2 * B2_total)
    total_lk = (1.0 - P_NL) * 100.0
    axial_lk = total_lk * (B2_axial / B2_total)
    return axial_lk, total_lk


def _hpmr_h_within_trained_range(n_rings_per_assembly, n_rings_per_core,
                                 active_height):
    """True if the user's H falls inside the per-(N_A, N_C) trained
    H range (5 % tolerance)."""
    _load()
    df = _train_df.dropna(subset=['H'])
    df = df[df['LT'] > 0]
    pair_df = df[(df['NA'] == n_rings_per_assembly) & (df['NC'] == n_rings_per_core)]
    if len(pair_df) == 0:
        # Fall back to the closest trained (NA, NC) pair
        pairs = df.groupby(['NA', 'NC']).size().index.tolist()
        if not pairs:
            return False
        nearest = min(pairs,
                      key=lambda p: abs(p[0] - n_rings_per_assembly) +
                                    abs(p[1] - n_rings_per_core))
        pair_df = df[(df['NA'] == nearest[0]) & (df['NC'] == nearest[1])]
    h_min, h_max = pair_df['H'].min(), pair_df['H'].max()
    return (h_min * 0.95) <= active_height <= (h_max * 1.05)


def get_hpmr_leakage(n_rings_per_assembly, n_rings_per_core, active_height,
                     enrichment, power_mwt,
                     active_radius_cm, radial_reflector_cm, axial_reflector_cm):
    """Return (axial_pct, total_pct, source) for HPMR. Source is
    'interpolated' for in-range queries, 'physics' for out-of-range."""
    if _hpmr_h_within_trained_range(n_rings_per_assembly, n_rings_per_core,
                                    active_height):
        ax = get_hpmr_axial_leakage_pct(n_rings_per_assembly, n_rings_per_core,
                                         active_height, enrichment, power_mwt)
        tot = get_hpmr_total_leakage_pct(n_rings_per_assembly, n_rings_per_core,
                                          active_height, enrichment, power_mwt)
        return ax, tot, 'interpolated'

    ax, tot = _hpmr_physics_leakage(active_radius_cm, active_height,
                                    radial_reflector_cm, axial_reflector_cm)
    return ax, tot, 'physics'
