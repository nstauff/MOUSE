# Copyright 2025 Battelle Energy Alliance, LLC
# Released under the MIT License.
"""
MOUSE Streamlit Web App
Microreactor Online Unified Simulation Engine — cost estimation without OpenMC.

Run from the MOUSE repo root:
    streamlit run webapp/app.py
"""

# ---------------------------------------------------------------------------
# Ensure the MOUSE repo root is in sys.path so all MOUSE modules resolve.
# ---------------------------------------------------------------------------
import sys
import os

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# ---------------------------------------------------------------------------
# IMPORTANT: Stub openmc and watts BEFORE any MOUSE import.
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock


class _MaterialStub:
    def __init__(self, name=None, temperature=None):
        self.name = name
        self.temperature = temperature
        self.density = 0.0

    def set_density(self, units, value):
        self.density = value

    def add_nuclide(self, *args, **kwargs):
        pass

    def add_element(self, *args, **kwargs):
        pass

    def add_s_alpha_beta(self, *args, **kwargs):
        pass

    @staticmethod
    def mix_materials(materials, fractions, method, name=None):
        result = _MaterialStub(name=name)
        try:
            result.density = sum(m.density * f for m, f in zip(materials, fractions))
        except Exception:
            result.density = 0.0
        return result


class _MaterialsStub:
    def append(self, mat):
        pass

    def extend(self, mats):
        pass


class _OpenMCStub(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Material = _MaterialStub
        self.Materials = _MaterialsStub


_openmc_stub = _OpenMCStub()
_openmc_stub.Material = _MaterialStub
_openmc_stub.Materials = _MaterialsStub

for _mod in ['openmc', 'openmc.deplete', 'openmc.mgxs']:
    sys.modules[_mod] = _openmc_stub

sys.modules['watts'] = MagicMock()

# ---------------------------------------------------------------------------
# Standard imports (after stubs are in place)
# ---------------------------------------------------------------------------
import io
import math
import sqlite3
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from st_cookies_manager import EncryptedCookieManager

from reactor_config import build_params, SubcriticalError, ShortLifetimeError, ESCALATION_YEAR
from webapp.fuel_lifetime_estimator import (
    get_ltmr_keff_curve,
    get_ltmr_peaking_factor,
    get_ltmr_leakage,
)
from webapp.gcmr_fuel_lifetime_estimator import (
    get_gcmr_peaking_factor,
    get_gcmr_leakage,
)
from cost.cost_estimation import bottom_up_cost_estimate, transform_dataframe
from cost.cost_drivers import cost_drivers_estimate, is_double_digit_excluding_multiples_of_10, get_detailed_driver_rows

# ---------------------------------------------------------------------------
# Performance patches: cache Excel reads that would otherwise repeat on every run.
# ---------------------------------------------------------------------------
import functools
import cost.cost_escalation as _ce

_orig_inflation = _ce.calculate_inflation_multiplier

@functools.lru_cache(maxsize=512)
def _cached_inflation_multiplier(file_path, base_dollar_year, cost_type, escalation_year):
    return _orig_inflation(file_path, base_dollar_year, cost_type, escalation_year)

_ce.calculate_inflation_multiplier = _cached_inflation_multiplier

import pandas as _pd_orig
_orig_read_excel = _pd_orig.read_excel

@functools.lru_cache(maxsize=32)
def _cached_read_excel(file_path, sheet_name):
    return _orig_read_excel(file_path, sheet_name=sheet_name)

def _patched_read_excel(file_path, sheet_name=0, **kwargs):
    if isinstance(file_path, str) and file_path.endswith('Cost_Database.xlsx') and not kwargs:
        return _cached_read_excel(file_path, sheet_name).copy()
    return _orig_read_excel(file_path, sheet_name=sheet_name, **kwargs)

_pd_orig.read_excel = _patched_read_excel
_ce.pd.read_excel = _patched_read_excel

# ---------------------------------------------------------------------------
# Reactor metadata: full names and design images
# ---------------------------------------------------------------------------
_REACTOR_LABELS = {
    'LTMR': 'Liquid Metal Microreactor (LTMR)',
    'GCMR': 'Gas Cooled Microreactor (GCMR)',
    'HPMR': 'Heat Pipe Microreactor (HPMR)',
}
_LABEL_TO_KEY = {v: k for k, v in _REACTOR_LABELS.items()}

_ASSETS = os.path.join(_repo_root, 'assets', 'Ref_openmc_2d_designs')

_REACTOR_IMAGES = {
    'LTMR': {
        'main': (
            os.path.join(_ASSETS, 'LTMR_core.png'),
            'LTMR core cross-section — hexagonal arrangement of TRIGA-type U-ZrH fuel '
            'pins and ZrH moderator pins cooled by NaK liquid metal, surrounded by a '
            'graphite radial reflector with control drums.',
        ),
        'details': [
            (
                os.path.join(_ASSETS, 'LTMR_fuel_pin_universe.png'),
                'LTMR fuel pin cross-section — from center outward: zirconium cladding, '
                'gap, U-ZrH fuel meat, gap, and SS304 outer cladding.',
            ),
            (
                os.path.join(_ASSETS, 'LTMR_moderator_pin_universe.png'),
                'LTMR moderator pin cross-section — ZrH hydrogen moderator encased in '
                'SS304 cladding, interspersed between fuel pins to thermalize neutrons.',
            ),
        ],
    },
    'GCMR': {
        # The main image is overridden at runtime with the per-(N_A, N_C)
        # core PNG; this static entry is the fallback when the per-pair
        # image isn't available.
        'main': (
            os.path.join(_ASSETS, 'GCMR_Core.png'),
            'GCMR core cross-section — hexagonal fuel assemblies containing TRISO fuel '
            'compacts arranged in a honeycomb pattern, cooled by helium gas flowing '
            'through dedicated coolant channels, with graphite reflector and control drums.',
        ),
        # The fuel assembly entry below is overridden at runtime with the
        # per-N_A image. The TRISO particle and zoomed fuel-assembly
        # images don't depend on geometry — kept static.
        'details': [
            (
                os.path.join(_ASSETS, 'GCMR_Fuel Assembly.png'),
                'GCMR fuel assembly cross-section — TRISO fuel compacts, helium coolant '
                'channels, and ZrH moderator booster pins embedded in a graphite matrix.',
            ),
            (
                os.path.join(_ASSETS, 'GCMR_fuel_assembly_zoomed.png'),
                'GCMR fuel assembly zoomed — individual TRISO particles visible within '
                'the graphite fuel compact at the target packing fraction.',
            ),
            (
                os.path.join(_ASSETS, 'GCMR_TRISO_particle.png'),
                'TRISO fuel particle — multi-layer design with UN fuel kernel, buffer '
                'graphite, inner PyC, SiC pressure vessel, and outer PyC coating that '
                'retains fission products up to ~1600 °C.',
            ),
        ],
    },
    'HPMR': {
        'main': (
            os.path.join(_ASSETS, 'HPMR_core.png'),
            'HPMR core cross-section — monolithic graphite/metal core with hexagonal '
            'fuel assemblies and embedded alkali-metal heat pipes that passively transfer '
            'heat to the secondary side, with graphite reflector and control drums.',
        ),
        'details': [
            (
                os.path.join(_ASSETS, 'HPMR_fuel_assembly.png'),
                'HPMR fuel assembly cross-section — TRISO fuel pins and heat pipes '
                'arranged in a hexagonal pattern within the graphite monolith block.',
            ),
            (
                os.path.join(_ASSETS, 'HPMR_fuel_pin_universe.png'),
                'HPMR fuel pin cross-section — homogenized TRISO fuel region surrounded '
                'by a thin helium gap within the monolith.',
            ),
            (
                os.path.join(_ASSETS, 'HPMR_heatpipe_universe.png'),
                'HPMR heat pipe cross-section — working fluid region and outer cladding '
                'that passively carries heat from the core to the power conversion system '
                'with no moving parts.',
            ),
        ],
    },
}

# ---------------------------------------------------------------------------
# Cached cost estimate (module-level so cache persists across reruns)
# ---------------------------------------------------------------------------
# Aspect-ratio bounds for the Active Height slider:
#     0.5 ≤ H / D ≤ 2.0
# where D is the active core diameter (= 2 × active_radius, no reflector).
# H is the active fuel height. Reflectors are excluded from both.
ASPECT_RATIO_MIN = 0.5
ASPECT_RATIO_MAX = 2.0

# LTMR per-N geometry:
#  - ACTIVE_R    = Core Radius − Reflector Thickness  (≈ 2.836·N − 1.136 cm)
#  - DIAMETER_CM = 2 × ACTIVE_R   (active core diameter, NO reflector)
# Trained N values come directly from the parametric study Excel; intermediate
# N values are linearly interpolated from the trained table — adequate for
# slider display, but flagged in the UI as interpolated.
LTMR_TRAINED_N = {10, 12, 14, 18, 24}
LTMR_N_TO_ACTIVE_RADIUS_CM = {
    n: round(2.836 * n - 1.136, 1)
    for n in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
}
LTMR_N_TO_DIAMETER_CM = {
    n: int(round(2 * ar))
    for n, ar in LTMR_N_TO_ACTIVE_RADIUS_CM.items()
}


def _ltmr_diameter_label(n, d):
    star = '' if n in LTMR_TRAINED_N else ' *'
    return f"{d} cm  (N={n}){star}"


LTMR_DIAMETER_LABELS = [_ltmr_diameter_label(n, d)
                        for n, d in LTMR_N_TO_DIAMETER_CM.items()]
LTMR_DIAMETER_LABEL_TO_N = {label: n
                            for n, label in zip(LTMR_N_TO_DIAMETER_CM.keys(),
                                                LTMR_DIAMETER_LABELS)}

# GCMR per-(N_A, N_C) geometry. Formulas (verified against parametric study):
#     Assembly_FTF(N_A) = 2.25 × (N_A − 1) × √3
#     Reflector(N_A)    = Assembly_FTF / 2
#     Active_Radius     = Assembly_FTF × N_C
#     Core_Radius       = Active_Radius + Reflector
#     Diameter          = 2 × Core_Radius
import math as _math
GCMR_TRAINED_PAIRS = {(4, 3), (5, 3), (6, 4), (6, 5), (7, 5)}
GCMR_NA_VALUES = [4, 5, 6, 7]
GCMR_NC_VALUES = [3, 4, 5]


def _gcmr_assembly_ftf(n_a):
    return 2.25 * (n_a - 1) * _math.sqrt(3.0)


def _gcmr_active_radius(n_a, n_c):
    return _gcmr_assembly_ftf(n_a) * n_c


def _gcmr_diameter_cm(n_a, n_c):
    ftf = _gcmr_assembly_ftf(n_a)
    return 2.0 * (ftf * n_c + ftf / 2.0)


# Build the 12-pair lookup of ACTIVE core diameter (no reflector),
# sorted ascending so the slider goes small → large left-to-right.
GCMR_PAIR_TO_DIAMETER_CM = {}
for _na in GCMR_NA_VALUES:
    for _nc in GCMR_NC_VALUES:
        GCMR_PAIR_TO_DIAMETER_CM[(_na, _nc)] = int(round(2 * _gcmr_active_radius(_na, _nc)))
GCMR_PAIR_TO_DIAMETER_CM = dict(sorted(GCMR_PAIR_TO_DIAMETER_CM.items(),
                                       key=lambda kv: kv[1]))


def _gcmr_diameter_label(na, nc, d):
    star = '' if (na, nc) in GCMR_TRAINED_PAIRS else ' *'
    return f"{d} cm  (N_A={na}, N_C={nc}){star}"


GCMR_DIAMETER_LABELS = [_gcmr_diameter_label(na, nc, d)
                        for (na, nc), d in GCMR_PAIR_TO_DIAMETER_CM.items()]
GCMR_DIAMETER_LABEL_TO_PAIR = {
    label: pair
    for pair, label in zip(GCMR_PAIR_TO_DIAMETER_CM.keys(),
                           GCMR_DIAMETER_LABELS)
}

# Mass-scaling constant derived from the GCMR parametric study reference
# row (N_A=6, N_C=5, H=215 cm, E=0.1975, P=1 MWt).  Total uranium mass per
# unit of [F_A × F_C × H] volume index.
GCMR_G_PER_VOLUME_INDEX = 0.5776


@st.cache_data(show_spinner=False)
def _run_estimate(reactor_type, power_mwt, enrichment, interest_rate, discount_rate,
                  construction_duration, debt_to_equity, operation_mode,
                  emergency_shutdowns, startup_duration, startup_duration_refueling,
                  tax_credit_type, tax_credit_value, plant_lifetime,
                  n_rings_per_assembly=None, active_height=None,
                  n_assembly_rings=None, n_core_rings=None):
    overrides = {
        'Interest Rate': interest_rate,
        'Discount Rate': discount_rate,
        'Construction Duration': construction_duration,
        'Debt To Equity Ratio': debt_to_equity,
        'Escalation Year': ESCALATION_YEAR,
        'Operation Mode': operation_mode,
        'Emergency Shutdowns Per Year': emergency_shutdowns,
        'Startup Duration after Emergency Shutdown': startup_duration,
        'Startup Duration after Refueling': startup_duration_refueling,
        'Levelization Period': plant_lifetime,
    }
    if tax_credit_type == 'PTC':
        overrides['PTC credit value'] = tax_credit_value
        overrides['PTC credit period'] = 10
        overrides['Tax Rate'] = 0.21
    elif tax_credit_type == 'ITC':
        overrides['ITC credit level'] = tax_credit_value

    p = build_params(reactor_type, power_mwt, enrichment, overrides,
                     n_rings_per_assembly=n_rings_per_assembly,
                     active_height=active_height,
                     n_assembly_rings=n_assembly_rings,
                     n_core_rings=n_core_rings)
    raw_df = bottom_up_cost_estimate('cost/Cost_Database.xlsx', p)
    enriched_df, detailed_sorted_df = cost_drivers_estimate(raw_df, p)
    return transform_dataframe(enriched_df), enriched_df, detailed_sorted_df, p


# Sweep NOAK Unit Number across a list of deployment scales for the
# "Costs in perspective" plot.  Each call to bottom_up_cost_estimate uses
# the same Number of Samples (10 — set inside reactor_config.py) as the
# headline LCOE card.  Cached on every input that influences cost.
@st.cache_data(show_spinner=False)
def _run_lcoe_sweep(reactor_type, power_mwt, enrichment, interest_rate, discount_rate,
                    construction_duration, debt_to_equity, operation_mode,
                    emergency_shutdowns, startup_duration, startup_duration_refueling,
                    tax_credit_type, tax_credit_value, plant_lifetime,
                    n_rings_per_assembly=None, active_height=None,
                    n_assembly_rings=None, n_core_rings=None,
                    units_tuple=(1, 5, 20, 50, 100)):
    base_overrides = {
        'Interest Rate': interest_rate,
        'Discount Rate': discount_rate,
        'Construction Duration': construction_duration,
        'Debt To Equity Ratio': debt_to_equity,
        'Escalation Year': ESCALATION_YEAR,
        'Operation Mode': operation_mode,
        'Emergency Shutdowns Per Year': emergency_shutdowns,
        'Startup Duration after Emergency Shutdown': startup_duration,
        'Startup Duration after Refueling': startup_duration_refueling,
        'Levelization Period': plant_lifetime,
    }
    if tax_credit_type == 'PTC':
        base_overrides['PTC credit value'] = tax_credit_value
        base_overrides['PTC credit period'] = 10
        base_overrides['Tax Rate'] = 0.21
        lcoe_account = 'LCOE with PTC'
    elif tax_credit_type == 'ITC':
        base_overrides['ITC credit level'] = tax_credit_value
        lcoe_account = 'LCOE (ITC-adjusted)'
    else:
        lcoe_account = 'LCOE'

    means, stds = [], []
    for N in units_tuple:
        overrides = dict(base_overrides)
        overrides['NOAK Unit Number'] = int(N)
        p = build_params(reactor_type, power_mwt, enrichment, overrides,
                         n_rings_per_assembly=n_rings_per_assembly,
                         active_height=active_height,
                         n_assembly_rings=n_assembly_rings,
                         n_core_rings=n_core_rings)
        raw_df = bottom_up_cost_estimate('cost/Cost_Database.xlsx', p)
        enriched_df, _ = cost_drivers_estimate(raw_df, p)
        # Read the 'LCOE' summary row's NOAK value directly from the
        # enriched dataframe, BEFORE transform_dataframe processes it.
        # The cost engine writes the LCOE row's value (in $/MWh) into
        # whichever column get_estimated_cost_column resolves for the
        # NOAK side (e.g. 'NOAK Estimated Cost ($2025M)').
        try:
            from cost.code_of_account_processing import get_estimated_cost_column as _gecc
            _noak_col = _gecc(enriched_df, 'N')
            _noak_std_col = _gecc(enriched_df, 'N std')
        except Exception:
            _noak_col = None
            _noak_std_col = None
        if (_noak_col is None
                or 'Account' not in enriched_df.columns
                or _noak_col not in enriched_df.columns):
            means.append(float('nan'))
            stds.append(float('nan'))
            continue
        _row = enriched_df[enriched_df['Account'] == lcoe_account]
        if _row.empty:
            means.append(float('nan'))
            stds.append(float('nan'))
            continue
        try:
            m = float(_row[_noak_col].iloc[0])
        except (TypeError, ValueError):
            m = float('nan')
        if _noak_std_col and _noak_std_col in enriched_df.columns:
            try:
                s = float(_row[_noak_std_col].iloc[0])
            except (TypeError, ValueError):
                s = 0.0
        else:
            s = 0.0
        means.append(m)
        stds.append(s)
    return list(units_tuple), means, stds


# Single-point cost-engine call for the Costs-in-Perspective plot.
# Used to add an intermediate anchor (e.g. NOAK Unit Number = 10)
# between the headline FOAK (N=1) and NOAK (N=user_setting).  Same
# extraction path as _run_estimate so the value matches the headline
# format exactly: bottom_up_cost_estimate -> cost_drivers_estimate ->
# transform_dataframe -> _get_mean_std.
@st.cache_data(show_spinner=False)
def _lcoe_at_noak_unit(reactor_type, power_mwt, enrichment, interest_rate, discount_rate,
                       construction_duration, debt_to_equity, operation_mode,
                       emergency_shutdowns, startup_duration, startup_duration_refueling,
                       tax_credit_type, tax_credit_value, plant_lifetime,
                       n_rings_per_assembly=None, active_height=None,
                       n_assembly_rings=None, n_core_rings=None,
                       noak_unit_number=10):
    """Returns (mean, std, per_account_df) for one NOAK Unit Number.

    The per_account_df includes Account, Account Title, FOAK and NOAK
    columns for all rows — useful for diagnosing which account drives
    a wild LCOE value.
    """
    overrides = {
        'Interest Rate': interest_rate,
        'Discount Rate': discount_rate,
        'Construction Duration': construction_duration,
        'Debt To Equity Ratio': debt_to_equity,
        'Escalation Year': ESCALATION_YEAR,
        'Operation Mode': operation_mode,
        'Emergency Shutdowns Per Year': emergency_shutdowns,
        'Startup Duration after Emergency Shutdown': startup_duration,
        'Startup Duration after Refueling': startup_duration_refueling,
        'Levelization Period': plant_lifetime,
        'NOAK Unit Number': int(noak_unit_number),
    }
    if tax_credit_type == 'PTC':
        overrides['PTC credit value'] = tax_credit_value
        overrides['PTC credit period'] = 10
        overrides['Tax Rate'] = 0.21
        lcoe_account = 'LCOE with PTC'
    elif tax_credit_type == 'ITC':
        overrides['ITC credit level'] = tax_credit_value
        lcoe_account = 'LCOE (ITC-adjusted)'
    else:
        lcoe_account = 'LCOE'
    p = build_params(reactor_type, power_mwt, enrichment, overrides,
                     n_rings_per_assembly=n_rings_per_assembly,
                     active_height=active_height,
                     n_assembly_rings=n_assembly_rings,
                     n_core_rings=n_core_rings)
    raw_df = bottom_up_cost_estimate('cost/Cost_Database.xlsx', p)
    enriched_df, _ = cost_drivers_estimate(raw_df, p)
    display_df = transform_dataframe(enriched_df)
    m, s = _get_mean_std(display_df, lcoe_account, 'NOAK')

    # Build a tidy per-account diagnostic frame.  Pulls 'Account',
    # 'Account Title', and any column starting with 'FOAK Estimated
    # Cost (' / 'NOAK Estimated Cost ('.
    _foak_cols = [c for c in display_df.columns if c.startswith('FOAK Estimated Cost (')]
    _noak_cols = [c for c in display_df.columns if c.startswith('NOAK Estimated Cost (')]
    _keep = [c for c in (['Account', 'Account Title']
                         + _foak_cols + _noak_cols) if c in display_df.columns]
    diag_df = display_df[_keep].copy() if _keep else None

    # Also pull a snapshot of the key params actually used for this
    # cost-engine call.  Reveals whether Construction Duration,
    # Power MWe, Capacity Factor etc. are what we expect.
    diag_params = {
        'NOAK Unit Number':            p.get('NOAK Unit Number'),
        'Power MWt':                   p.get('Power MWt'),
        'Power MWe':                   p.get('Power MWe'),
        'Thermal Efficiency':          p.get('Thermal Efficiency'),
        'Capacity Factor':             p.get('Capacity Factor'),
        'Construction Duration':       p.get('Construction Duration'),
        'Interest Rate':               p.get('Interest Rate'),
        'Discount Rate':               p.get('Discount Rate'),
        'Debt To Equity Ratio':        p.get('Debt To Equity Ratio'),
        'Levelization Period':         p.get('Levelization Period'),
        'Annual Electricity Production': p.get('Annual Electricity Production'),
    }

    return (float(m) if m == m else float('nan'),
            float(s) if s == s else 0.0,
            diag_df,
            diag_params)


# ---------------------------------------------------------------------------
# Anonymous visitor analytics
# ---------------------------------------------------------------------------
_ANALYTICS_DB_PATH = Path(_repo_root) / 'webapp' / 'analytics.db'
_ANALYTICS_COOKIE_NAME = 'anonymous_id'


def _get_cookie_manager():
    return EncryptedCookieManager(
        prefix='mouse_',
        password=st.secrets.get('COOKIE_PASSWORD', 'change-this-in-streamlit-secrets'),
    )


def _get_analytics_conn():
    conn = sqlite3.connect(_ANALYTICS_DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            anonymous_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            first_seen_utc TEXT NOT NULL,
            visit_time_utc TEXT NOT NULL,
            page_name TEXT,
            reactor_type TEXT,
            user_agent TEXT,
            language TEXT
        )
    """)
    conn.commit()
    return conn


def _get_or_create_anonymous_id(cookies):
    anonymous_id = cookies.get(_ANALYTICS_COOKIE_NAME)
    if not anonymous_id:
        anonymous_id = str(uuid.uuid4())
        cookies[_ANALYTICS_COOKIE_NAME] = anonymous_id
        cookies.save()
    return anonymous_id


def _get_first_seen(conn, anonymous_id, now_utc):
    row = conn.execute(
        "SELECT MIN(first_seen_utc) FROM visits WHERE anonymous_id = ?",
        (anonymous_id,),
    ).fetchone()
    return row[0] if row and row[0] else now_utc


def _log_visit_once_per_session(conn, anonymous_id, reactor_type=None, page_name='main'):
    if 'analytics_session_id' not in st.session_state:
        st.session_state.analytics_session_id = str(uuid.uuid4())

    if 'analytics_visit_logged' not in st.session_state:
        st.session_state.analytics_visit_logged = False

    if st.session_state.analytics_visit_logged:
        return

    headers = dict(st.context.headers) if hasattr(st.context, 'headers') else {}
    user_agent = headers.get('User-Agent', '')
    language = headers.get('Accept-Language', '')
    now_utc = datetime.now(timezone.utc).isoformat()
    first_seen_utc = _get_first_seen(conn, anonymous_id, now_utc)

    conn.execute(
        """
        INSERT INTO visits (
            anonymous_id,
            session_id,
            first_seen_utc,
            visit_time_utc,
            page_name,
            reactor_type,
            user_agent,
            language
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            anonymous_id,
            st.session_state.analytics_session_id,
            first_seen_utc,
            now_utc,
            page_name,
            reactor_type,
            user_agent,
            language,
        ),
    )
    conn.commit()
    st.session_state.analytics_visit_logged = True


def _get_analytics_summary(conn):
    total_visits = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
    unique_visitors = conn.execute(
        "SELECT COUNT(DISTINCT anonymous_id) FROM visits"
    ).fetchone()[0]
    returning_visitors = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT anonymous_id
            FROM visits
            GROUP BY anonymous_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    visits_per_person = pd.read_sql_query(
        """
        SELECT
            anonymous_id,
            COUNT(*) AS visit_count,
            MIN(first_seen_utc) AS first_seen_utc,
            MAX(visit_time_utc) AS last_visit_utc,
            MAX(COALESCE(reactor_type, '')) AS last_reactor_type
        FROM visits
        GROUP BY anonymous_id
        ORDER BY last_visit_utc DESC
        """,
        conn,
    )

    recent_visits = pd.read_sql_query(
        """
        SELECT
            anonymous_id,
            visit_time_utc,
            reactor_type,
            page_name
        FROM visits
        ORDER BY visit_time_utc DESC
        LIMIT 25
        """,
        conn,
    )

    return total_visits, unique_visitors, returning_visitors, visits_per_person, recent_visits


def _render_analytics_sidebar(conn):
    total_visits, unique_visitors, returning_visitors, visits_per_person, recent_visits = _get_analytics_summary(conn)

    with st.sidebar.expander('📈 Anonymous Visitor Analytics'):
        c1, c2 = st.columns(2)
        c1.metric('Total visits', total_visits)
        c2.metric('Unique visitors', unique_visitors)

        c3, c4 = st.columns(2)
        c3.metric('Returning', returning_visitors)
        c4.metric(
            'Avg visits/user',
            f"{(total_visits / unique_visitors):.2f}" if unique_visitors else '0.00'
        )

        st.caption('Visits per anonymous visitor')
        st.dataframe(
            visits_per_person,
            use_container_width=True,
            hide_index=True,
            height=220,
        )

        st.caption('Most recent visits')
        st.dataframe(
            recent_visits,
            use_container_width=True,
            hide_index=True,
            height=220,
        )

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _get_mean_std(df, account, which='FOAK'):
    row = df[df['Account'] == account]
    if row.empty:
        return float('nan'), float('nan')
    mean_prefix = 'FOAK Estimated Cost (' if which == 'FOAK' else 'NOAK Estimated Cost ('
    std_prefix  = 'FOAK Estimated Cost std (' if which == 'FOAK' else 'NOAK Estimated Cost std ('
    mean_cols = [c for c in df.columns if c.startswith(mean_prefix)]
    std_cols  = [c for c in df.columns if c.startswith(std_prefix)]
    try:
        mean = float(row[mean_cols[0]].iloc[0]) if mean_cols else float('nan')
    except (TypeError, ValueError):
        mean = float('nan')
    try:
        std = float(row[std_cols[0]].iloc[0]) if std_cols else float('nan')
    except (TypeError, ValueError):
        std = float('nan')
    return mean, std


def _fmt_cost(mean, std):
    if math.isnan(mean):
        return 'N/A'
    m = round(mean / 1e6)
    if math.isnan(std) or std == 0:
        return f'${m}M'
    lo = round((mean - std) / 1e6)
    hi = round((mean + std) / 1e6)
    return f'${lo}M – ${hi}M'


def _fmt_lcoe(mean, std):
    if math.isnan(mean):
        return 'N/A'
    m = int(round(mean))
    if math.isnan(std) or std == 0:
        return f'${m}/MW<sub>e</sub>h'
    lo = int(round(mean - std))
    hi = int(round(mean + std))
    return f'${lo} – ${hi}/MW<sub>e</sub>h'


def _fmt_lcoh(mean, std):
    if math.isnan(mean):
        return 'N/A'
    m = int(round(mean))
    if math.isnan(std) or std == 0:
        return f'${m}/MW<sub>t</sub>h'
    lo = int(round(mean - std))
    hi = int(round(mean + std))
    return f'${lo} – ${hi}/MW<sub>t</sub>h'


def _get_lcof(df, which='FOAK'):
    mean_col = 'FOAK LCOE' if which == 'FOAK' else 'NOAK LCOE'
    std_col  = 'FOAK LCOE_std' if which == 'FOAK' else 'NOAK LCOE_std'
    if mean_col not in df.columns:
        return float('nan'), float('nan')
    mask = df['Account'].isin([25, 80])
    mean_vals = pd.to_numeric(df.loc[mask, mean_col], errors='coerce').dropna()
    mean = float(mean_vals.sum()) if len(mean_vals) > 0 else float('nan')
    if std_col not in df.columns:
        return mean, float('nan')
    std_vals = pd.to_numeric(df.loc[mask, std_col], errors='coerce').fillna(0)
    std = float(np.sqrt((std_vals ** 2).sum()))
    return mean, std


def _fmt_table_val(x):
    if x == '-' or x is None or x == '':
        return x
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v == 0:
        return '0'
    sign = '-' if v < 0 else ''
    a = abs(v)
    if a >= 1e7:
        return f'{sign}{round(a / 1e6):.0f}M'
    elif a >= 1e6:
        return f'{sign}{round(a / 1e6, 1)}M'
    elif a >= 1e5:
        return f'{sign}{round(a / 1e4) * 10:.0f}K'
    elif a >= 1e4:
        return f'{sign}{round(a / 1e3):.0f}K'
    elif a >= 1e3:
        return f'{sign}{round(a / 1e3, 1)}K'
    else:
        return f'{sign}{int(round(a))}'


# ---------------------------------------------------------------------------
# HTML/CSS card helpers
# ---------------------------------------------------------------------------
_CARD_COLORS = {
    'occ':  '#7c3aed',
    'tci':  '#0369a1',
    'lcoe': '#0891b2',
    'lcoh': '#059669',
    'lcof': '#d97706',
}

def _kpi_card(col, title, foak_val, noak_val, color='#1B4F8C'):
    col.markdown(
        f'''<div style="background:white;border-radius:14px;padding:1.1rem 1.25rem;
                        box-shadow:0 2px 12px rgba(0,0,0,0.07);border-top:3px solid {color};
                        min-height:110px;">
              <div style="font-size:0.68rem;font-weight:700;color:#9ca3af;
                          text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.65rem;">{title}</div>
              <div style="display:flex;align-items:baseline;gap:0.45rem;margin-bottom:0.45rem;">
                <span style="background:#fff3ed;color:#c84b1e;font-size:0.6rem;font-weight:800;
                             padding:0.12rem 0.38rem;border-radius:4px;letter-spacing:0.05em;
                             flex-shrink:0;">FOAK</span>
                <span style="font-size:1rem;font-weight:700;color:#111827;">{foak_val}</span>
              </div>
              <div style="height:1px;background:#f3f4f6;margin:0.3rem 0;"></div>
              <div style="display:flex;align-items:baseline;gap:0.45rem;margin-top:0.45rem;">
                <span style="background:#eff6ff;color:#1d4ed8;font-size:0.6rem;font-weight:800;
                             padding:0.12rem 0.38rem;border-radius:4px;letter-spacing:0.05em;
                             flex-shrink:0;">NOAK</span>
                <span style="font-size:1rem;font-weight:700;color:#111827;">{noak_val}</span>
              </div>
            </div>''',
        unsafe_allow_html=True,
    )


def _make_side_view_figure(diameter_cm, active_height_cm,
                           axial_reflector_cm, radial_reflector_cm):
    """Return a matplotlib figure showing a to-scale side view of the
    reactor cylinder.

    Layout matches the cross-section image directly above it: the plot
    box is placed in the LEFT ~55% of the figure (the right ~35% is left
    empty to mirror where the cross-section's material legend sits), and
    its xlim spans the same range the cross-section uses (= ±ceil(D/2)).
    The axis box dimensions are picked so 1 cm in x equals 1 cm in y,
    without invoking aspect='equal' — that way the rectangle width
    (= D) renders at exactly the same horizontal pixel range as the
    circle in the cross-section above.
    """
    total_h = active_height_cm + 2 * axial_reflector_cm
    active_d = diameter_cm - 2 * radial_reflector_cm

    # Match the cross-section's xlim/ylim convention (ceil to integer)
    half_d_int = max(1, int(np.ceil(diameter_cm / 2.0)))
    half_h_int = max(1, int(np.ceil(total_h / 2.0)))

    # Axis box dimensions in inches, sized so cm/inch is the same in x and y.
    box_w = 4.0
    box_h = box_w * (half_h_int / half_d_int)   # H/D aspect, preserved exactly
    # Cap very tall/short reactors so the figure stays a reasonable size.
    box_h = max(0.6, min(8.0, box_h))

    # The plot occupies the same x-range as the cross-section's plot
    # box (≈ 13%–62% of the column width — between the leftmost and
    # rightmost x-tick of the cross-section image). The right ~38% is
    # empty (mirroring the cross-section legend area), and the left
    # ~13% is empty (matching the cross-section's y-axis label margin).
    ax_left_frac, ax_width_frac  = 0.13, 0.49
    ax_bot_frac,  ax_height_frac = 0.13, 0.72
    fig_w = box_w / ax_width_frac
    fig_h = box_h / ax_height_frac + 0.4   # +0.4 inches for title/labels

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([ax_left_frac, ax_bot_frac, ax_width_frac, ax_height_frac])

    # Outer envelope (radial reflector) — full diameter × full height
    outer = mpatches.Rectangle((-diameter_cm / 2.0, -total_h / 2.0),
                               diameter_cm, total_h,
                               facecolor='#fde68a', edgecolor='#92400e', lw=1.0)
    ax.add_patch(outer)

    # Active core — active diameter × active height, centred
    core = mpatches.Rectangle((-active_d / 2.0, -active_height_cm / 2.0),
                              active_d, active_height_cm,
                              facecolor='#fca5a5', edgecolor='#7f1d1d', lw=1.0)
    ax.add_patch(core)

    # Dimension labels — active labels are placed close to the inner
    # (red) core rectangle; total labels are placed next to the outer
    # (yellow) envelope. annotation_clip=False so labels can spill
    # outside the axis box.
    #
    # H_active: just to the RIGHT of the inner core, slightly ABOVE the
    # vertical centre. H_total: outside the outer envelope on the right,
    # slightly BELOW centre — staggered so the two labels don't overlap.
    ax.annotate(f'H_active = {active_height_cm:.0f} cm',
                xy=(active_d / 2.0 + 0.01 * half_d_int, +half_h_int * 0.30),
                ha='left', va='center', fontsize=7, fontweight='bold',
                color='#7f1d1d', annotation_clip=False)
    ax.annotate(f'H_total = {total_h:.0f} cm',
                xy=(half_d_int * 1.04, -half_h_int * 0.30),
                ha='left', va='center', fontsize=8, fontweight='bold',
                color='#92400e', annotation_clip=False)
    #
    # D_active: just BELOW the inner core, in the axial reflector gap.
    # D_total: outside the outer envelope at the bottom.
    ax.annotate(f'D_active = {active_d:.0f} cm',
                xy=(0, -active_height_cm / 2.0 - 0.01 * half_h_int),
                ha='center', va='top', fontsize=7, fontweight='bold',
                color='#7f1d1d', annotation_clip=False)
    ax.annotate(f'D_total = {diameter_cm:.0f} cm',
                xy=(0, -half_h_int * 1.08),
                ha='center', va='top', fontsize=8, fontweight='bold',
                color='#92400e', annotation_clip=False)

    ax.set_xlim(-half_d_int, half_d_int)
    ax.set_ylim(-half_h_int, half_h_int)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('Side View (to scale)', fontsize=10, fontweight='bold')

    return fig


def _materials_section(reactor_type, params):
    """Render a 'Materials & Components' panel that lists the basic
    materials of the reactor (fuel, moderator, reflector, drums, …)
    plus, for LTMR, the per-assembly fuel and moderator pin counts.
    All values are read directly from `params` — no interpolation."""
    def _pretty(name):
        """Replace underscores with spaces so material names like
        'UZrH_alloy' read as 'UZrH alloy' in the displayed table."""
        if not isinstance(name, str):
            return name
        return name.replace('_', ' ')

    rows = []
    # Fuel — for LTMR (UZrH alloy) we know the U weight fraction, so
    # show it next to the material name.
    _fuel_str = _pretty(params.get('Fuel', '—'))
    if reactor_type == 'LTMR' and 'U_met_wo' in params:
        _u_wo = float(params['U_met_wo']) * 100.0
        _fuel_str = f"{_fuel_str}  ({_u_wo:.0f} wt% U)"
    rows.append(('Fuel', _fuel_str))

    rows.append(('Moderator', _pretty(params.get('Moderator', '—'))))
    if params.get('Moderator Booster Materials'):
        rows.append(('Moderator booster',
                     ', '.join(_pretty(m) for m in params['Moderator Booster Materials'])))
    if reactor_type == 'HPMR':
        rows.append(('Cooling device', _pretty(params.get('Cooling Device', 'Heat pipes'))))
    else:
        rows.append(('Coolant', _pretty(params.get('Coolant', '—'))))
    rows.append(('Radial reflector', _pretty(params.get('Radial Reflector', '—'))))
    rows.append(('Axial reflector',  _pretty(params.get('Axial Reflector',  '—'))))
    rows.append(('Control drum absorber',  _pretty(params.get('Control Drum Absorber',  '—'))))
    rows.append(('Control drum reflector', _pretty(params.get('Control Drum Reflector', '—'))))

    # LTMR is a single-assembly core — no per-assembly distinction needed.
    if reactor_type == 'LTMR':
        if 'Fuel Pin Count' in params:
            rows.append(('Number of fuel pins',
                         f"{int(params['Fuel Pin Count']):,}"))
        if 'Moderator Pin Count' in params:
            rows.append(('Number of moderator pins',
                         f"{int(params['Moderator Pin Count']):,}"))

    body = ''.join(
        f'<tr>'
        f'<td style="padding:0.32rem 1rem 0.32rem 0;color:#475569;font-weight:600;'
        f'white-space:nowrap;">{k}</td>'
        f'<td style="padding:0.32rem 0;color:#0f172a;font-weight:500;">{v}</td>'
        f'</tr>'
        for k, v in rows
    )
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;'
        'letter-spacing:0.09em;margin-bottom:0.45rem;">Materials &amp; Components</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'''<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;
                        padding:0.9rem 1.15rem;margin-bottom:0.85rem;">
              <table style="width:100%;font-size:0.86rem;border-collapse:collapse;">
                <tbody>{body}</tbody>
              </table>
            </div>''',
        unsafe_allow_html=True,
    )


def _info_card(col, title, value, subtitle='', accent='#16a34a', bg='#f0fdf4', border='#bbf7d0'):
    sub_html = f'<div style="font-size:0.7rem;color:#6b7280;margin-top:0.2rem;">{subtitle}</div>' if subtitle else ''
    col.markdown(
        f'''<div style="background:{bg};border:1px solid {border};border-radius:14px;
                        padding:1.1rem 1.25rem;box-shadow:0 2px 8px rgba(0,0,0,0.04);min-height:80px;">
              <div style="font-size:0.68rem;font-weight:700;color:{accent};
                          text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.35rem;">{title}</div>
              <div style="font-size:1.35rem;font-weight:800;color:#111827;line-height:1.2;">{value}</div>
              {sub_html}
            </div>''',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── App background ── */
.stApp { background: #eef2f8 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(175deg, #0b1f3a 0%, #1a3d66 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown strong {
    color: #93c5fd !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] .stCaption p {
    color: #475569 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 0.6rem 0 !important;
}
section[data-testid="stSidebar"] .stTooltipHoverTarget svg.icon {
    stroke: #93c5fd !important;
    stroke-width: 2.25 !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #e05c2b 0%, #b84520 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 15px rgba(224,92,43,0.35) !important;
    letter-spacing: 0.02em !important;
    width: 100%;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #f06a35 0%, #c9531e 100%) !important;
    box-shadow: 0 6px 20px rgba(224,92,43,0.5) !important;
    transform: translateY(-1px);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0.25rem;
    border-bottom: 2px solid #dde3ee;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-radius: 8px 8px 0 0;
    color: #64748b;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0.55rem 1.2rem;
    margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1B4F8C !important;
    border-top: 1px solid #dde3ee !important;
    border-left: 1px solid #dde3ee !important;
    border-right: 1px solid #dde3ee !important;
    border-bottom: 2px solid white !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: white;
    border-radius: 0 12px 12px 12px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: white;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1B4F8C, #1565c0) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 10px rgba(27,79,140,0.3) !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1565c0, #1976d2) !important;
    box-shadow: 0 5px 15px rgba(27,79,140,0.45) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: #1B4F8C !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
"""

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title='MOUSE — Microreactor Cost Estimator',
    page_icon='⚛',
    layout='wide',
)

cookies = _get_cookie_manager()
if not cookies.ready():
    st.stop()

analytics_conn = _get_analytics_conn()
anonymous_id = _get_or_create_anonymous_id(cookies)

# ---------------------------------------------------------------------------
# Main app — wrapped in analytics tracker
# ---------------------------------------------------------------------------
with streamlit_analytics.track():

    st.markdown(f'<style>{_CSS}</style>', unsafe_allow_html=True)

    # ── Sidebar inputs ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:0.5rem 0 1rem 0;">'
            '<span style="font-size:2.2rem;">⚛</span>'
            '<div style="color:white;font-weight:800;font-size:1.1rem;letter-spacing:0.05em;margin-top:0.2rem;">MOUSE</div>'
            '<div style="color:#64748b;font-size:0.65rem;letter-spacing:0.06em;text-transform:uppercase;line-height:1.4;">'
            'Microreactor Optimization<br>Using Simulation &amp; Economics</div>'
            '<div style="color:#94a3b8;font-size:0.72rem;font-weight:600;margin-top:0.5rem;letter-spacing:0.02em;">'
            '🏛 Idaho National Laboratory</div>'
            '<div style="margin-top:0.6rem;">'
            '<a href="https://github.com/IdahoLabResearch/MOUSE" target="_blank" '
            'style="display:inline-block;background:#24292e;color:white;font-size:0.72rem;'
            'font-weight:600;text-decoration:none;padding:0.3rem 0.8rem;border-radius:20px;'
            'border:1px solid #444;letter-spacing:0.02em;">&#9651; Open Source on GitHub</a>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown('**A — Reactor Design**')

        reactor_label = st.selectbox(
            'Reactor Type',
            options=list(_REACTOR_LABELS.values()),
            help='Select a microreactor design to estimate costs for.',
        )
        reactor_type = _LABEL_TO_KEY[reactor_label]

        _log_visit_once_per_session(analytics_conn, anonymous_id, reactor_type=reactor_type, page_name='main')

        enrichment = st.slider(
            'Enrichment (U-235 fraction)',
            min_value=0.05,
            max_value=0.1975,
            value=0.1975,
            step=0.0025,
            format='%.4f',
            help='U-235 enrichment fraction. Affects uranium masses and fuel lifetime via interpolation.',
        )
        st.caption(f'{enrichment * 100:.2f}% enriched')

        _power_defaults = {'LTMR': 20, 'GCMR': 15, 'HPMR': 7}
        _power_max = {'LTMR': 64, 'GCMR': 50, 'HPMR': 7}

        power_mwt = st.slider(
            'Thermal Power (MWt)',
            min_value=1,
            max_value=_power_max[reactor_type],
            value=_power_defaults[reactor_type],
            step=1,
            key=f'power_{reactor_type}',
            help='Thermal power output. Affects power-dependent params and fuel lifetime via interpolation.',
        )

        # LTMR / GCMR: extra geometry inputs (diameter + active height).
        n_rings_per_assembly = None  # LTMR only
        n_assembly_rings     = None  # GCMR only
        n_core_rings         = None  # GCMR only
        active_height        = None
        if reactor_type == 'LTMR':
            # Default to N=12 (95 cm), a mid-range trained geometry.
            _default_diameter_label = next(
                lbl for lbl in LTMR_DIAMETER_LABELS
                if LTMR_DIAMETER_LABEL_TO_N[lbl] == 12
            )
            _diameter_label = st.select_slider(
                'Active Core Diameter',
                options=LTMR_DIAMETER_LABELS,
                value=_default_diameter_label,
                key='ltmr_diameter',
                help=('Active core diameter (does NOT include the radial reflector). '
                      'Discrete values mapped to the number of fuel rings per '
                      'assembly. Values marked with * are interpolated between '
                      'trained geometries.'),
            )
            n_rings_per_assembly = LTMR_DIAMETER_LABEL_TO_N[_diameter_label]

            _ar_ltmr   = LTMR_N_TO_ACTIVE_RADIUS_CM[n_rings_per_assembly]
            _ad_ltmr   = LTMR_N_TO_DIAMETER_CM[n_rings_per_assembly]   # active diameter
            _h_min     = max(1, int(round(ASPECT_RATIO_MIN * _ad_ltmr)))
            _h_max     = int(round(ASPECT_RATIO_MAX * _ad_ltmr))
            _h_default = int(round(_ad_ltmr))   # H/D = 1.0

            active_height = st.slider(
                'Active Height (cm)',
                min_value=_h_min,
                max_value=_h_max,
                value=_h_default,
                step=1,
                key=f'ltmr_active_height_{n_rings_per_assembly}',
                help=(f'Active fuel height in cm. Bounds correspond to aspect ratio '
                      f'(H / D, where D is the active core diameter) between '
                      f'{ASPECT_RATIO_MIN} and {ASPECT_RATIO_MAX}. For this geometry '
                      f'the active core diameter is {_ad_ltmr} cm.'),
            )

        elif reactor_type == 'GCMR':
            # Default to (N_A=6, N_C=5), the reference GCMR design
            _default_label = next(
                lbl for lbl in GCMR_DIAMETER_LABELS
                if GCMR_DIAMETER_LABEL_TO_PAIR[lbl] == (6, 5)
            )
            _diameter_label = st.select_slider(
                'Active Core Diameter',
                options=GCMR_DIAMETER_LABELS,
                value=_default_label,
                key='gcmr_diameter',
                help=('Active core diameter (does NOT include the radial reflector). '
                      'Discrete values mapped to (Assembly Rings, Core Rings). Values '
                      'marked with * are interpolated between trained geometries.'),
            )
            n_assembly_rings, n_core_rings = GCMR_DIAMETER_LABEL_TO_PAIR[_diameter_label]

            _ar_gcmr   = _gcmr_active_radius(n_assembly_rings, n_core_rings)
            _ad_gcmr   = 2.0 * _ar_gcmr                                 # active diameter
            _h_min     = max(1, int(round(ASPECT_RATIO_MIN * _ad_gcmr)))
            _h_max     = int(round(ASPECT_RATIO_MAX * _ad_gcmr))
            _h_default = int(round(_ad_gcmr))   # H/D = 1.0

            active_height = st.slider(
                'Active Height (cm)',
                min_value=_h_min,
                max_value=_h_max,
                value=_h_default,
                step=1,
                key=f'gcmr_active_height_{n_assembly_rings}_{n_core_rings}',
                help=(f'Active fuel height in cm. Bounds correspond to aspect ratio '
                      f'(H / D, where D is the active core diameter) between '
                      f'{ASPECT_RATIO_MIN} and {ASPECT_RATIO_MAX}. For this geometry '
                      f'the active core diameter is {_ad_gcmr:.0f} cm.'),
            )

        st.divider()
        st.markdown('**B — Operation Parameters**')

        _OPERATION_MODE_LABELS = {
            'Remotely Monitored': 'Autonomous',
            'On-Site Staffed':    'Non-Autonomous',
        }
        operation_mode_label = st.selectbox(
            'Operation Mode',
            options=list(_OPERATION_MODE_LABELS.keys()),
            help=(
                '**Remotely Monitored:** Operators monitor the reactor remotely and are '
                'required on-site only for emergencies or shutdown.\n\n'
                '**On-Site Staffed:** Operators must be physically present in the control room 24/7.'
            ),
        )
        operation_mode = _OPERATION_MODE_LABELS[operation_mode_label]
        emergency_shutdowns = st.number_input(
            'Emergency Shutdowns per Year',
            min_value=0.0, max_value=10.0, value=2.0, step=0.1, format='%.1f',
            help='Expected number of unplanned emergency shutdowns per year.',
        )
        startup_duration = st.number_input(
            'Startup Duration after Emergency Shutdown (days)',
            min_value=1, max_value=365, value=21, step=1,
            help='Days required to restart the reactor after an emergency shutdown.',
        )
        startup_duration_refueling = st.number_input(
            'Startup Duration after Refueling (days)',
            min_value=1, max_value=365, value=14, step=1,
            help='Days required to restart the reactor after a scheduled refueling outage.',
        )

        st.divider()
        st.markdown('**C — Economic Parameters**')

        interest_rate = st.number_input(
            'Interest Rate (%)',
            min_value=2.0, max_value=15.0, value=7.0, step=0.5, format='%.1f',
            help=(
                'Annual cost of debt — the rate at which the project borrows money to finance construction. '
                'Used only to calculate interest expenses during construction (Account 62). '
                'Typical range: 2–15%. Nuclear projects commonly use 5–12%.'
            ),
        )
        discount_rate = st.number_input(
            'Discount Rate (%)',
            min_value=3.0, max_value=15.0, value=7.0, step=0.5, format='%.1f',
            help=(
                'Annual discount rate (Weighted Average Cost of Capital, WACC) — reflects the '
                'opportunity cost of capital and the time value of money. Used for LCOE/LCOH '
                'levelization and cost annualization. Should be >= interest rate. '
                'Typical range: 3–15%. Government/public projects: 3–7%; private nuclear: 8–15%.'
            ),
        )
        construction_duration = st.number_input(
            'Construction Duration (months)',
            min_value=1, max_value=120, value=12, step=1,
            help='Number of months from ground-break to commercial operation.',
        )
        debt_to_equity = st.slider(
            'Debt-to-Equity Ratio',
            min_value=0.0, max_value=5.0, value=1.0, step=0.1, format='%.1f',
            help='Ratio of debt to equity financing (e.g. 1.0 = equal debt and equity).',
        )
        plant_lifetime = st.number_input(
            'Plant Lifetime (years)',
            min_value=10, max_value=100, value=60, step=1,
            help=(
                'Economic lifetime of the plant used to levelize costs over time. '
                'A longer lifetime spreads capital costs over more years, reducing LCOE. '
                'Minimum: 10 years. Maximum: 100 years.'
            ),
        )

        st.markdown('**Government Subsidy (IRA Tax Credits)**')
        tax_credit_type = st.selectbox(
            'Tax Credit',
            options=['None', 'PTC', 'ITC'],
            index=0,
            help='None: no credit. PTC: Production Tax Credit (reduces LCOE). ITC: Investment Tax Credit (reduces OCC).',
        )
        tax_credit_value = None
        if tax_credit_type == 'PTC':
            tax_credit_value = st.selectbox(
                'PTC Credit Value ($/MWh)',
                options=[3.0, 3.3, 3.6, 15.0, 16.5, 18.0],
                index=3,
                format_func=lambda x: f'${x:.1f}/MWh',
                help='Total PTC value including any applicable IRA bonus multipliers.',
            )
        elif tax_credit_type == 'ITC':
            tax_credit_value = st.selectbox(
                'ITC Credit Level',
                options=[0.06, 0.30, 0.40, 0.50],
                index=1,
                format_func=lambda x: f'{x*100:.0f}%',
                help='ITC as a fraction of overnight capital cost (OCC).',
            )

        st.divider()
        run_button = st.button('⚡  Run Cost Estimate', type='primary', use_container_width=True)

        st.divider()
        st.markdown('**💬 Feedback**')
        st.caption('Help us improve MOUSE by sharing your thoughts.')
        st.link_button(
            '📝  Give Feedback',
            'https://qualtricsxm69xy9s7vm.qualtrics.com/jfe/form/SV_4Pb0vub9xCcsVV4',
            use_container_width=True,
        )

        if st.secrets.get("SHOW_ANALYTICS_PANEL", False):
            st.divider()
            _render_analytics_sidebar(analytics_conn)

    # ── Welcome banner ──────────────────────────────────────────────────────
    if not run_button:
        st.markdown(
            f'''<div style="background:linear-gradient(135deg,#0b1f3a 0%,#1B4F8C 55%,#1e6fa8 100%);
                           border-radius:18px;padding:3rem 3rem 2.8rem;color:white;
                           margin-bottom:1.5rem;position:relative;overflow:hidden;">
                 <div style="position:absolute;top:-60px;right:-60px;width:300px;height:300px;
                             border-radius:50%;background:rgba(255,255,255,0.04);"></div>
                 <div style="position:absolute;right:3rem;top:50%;transform:translateY(-50%);
                             font-size:7rem;opacity:0.1;line-height:1;">⚛</div>
                 <div style="display:flex;align-items:center;gap:0.6rem;flex-wrap:wrap;margin-bottom:1.1rem;">
                   <span style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.35);
                                 border-radius:20px;padding:0.35rem 1rem;font-size:0.82rem;font-weight:700;
                                 color:white;letter-spacing:0.02em;">
                     🏛 Idaho National Laboratory
                   </span>
                   <a href="https://github.com/IdahoLabResearch/MOUSE" target="_blank"
                      style="background:#24292e;border:1px solid rgba(255,255,255,0.25);
                             border-radius:20px;padding:0.35rem 1rem;font-size:0.82rem;font-weight:700;
                             color:white;text-decoration:none;letter-spacing:0.02em;">
                     &#9651; Open Source on GitHub &rarr; IdahoLabResearch/MOUSE
                   </a>
                 </div>
                 <h1 style="font-size:2.4rem;font-weight:800;margin:0 0 0.3rem;color:white;line-height:1.15;">
                   MOUSE
                 </h1>
                 <div style="font-size:1rem;font-weight:600;opacity:0.85;margin-bottom:0.9rem;color:white;">
                   Microreactor Optimization Using Simulation and Economics
                 </div>
                 <p style="font-size:0.92rem;opacity:0.75;margin:0 0 1.8rem;max-width:620px;color:white;line-height:1.6;">
                   MOUSE bridges nuclear microreactor design and economics by integrating core physics
                   simulations (OpenMC), simplified balance-of-plant calculations, and detailed bottom-up
                   cost estimation — enabling parametric optimization studies and uncertainty analysis
                   for both <strong style="color:white;">First-of-a-Kind (FOAK)</strong> and
                   <strong style="color:white;">Nth-of-a-Kind (NOAK)</strong> deployments.
                   Cost estimation correlations derive from the MARVEL project and supplementary literature.
                   All costs in <strong style="color:white;">{ESCALATION_YEAR} USD</strong>.
                 </p>
                 <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
                   <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                               border-radius:10px;padding:0.7rem 1.2rem;">
                     <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                                 opacity:0.6;margin-bottom:0.2rem;">Reactor Types</div>
                     <div style="font-weight:700;font-size:0.88rem;">LTMR · GCMR · HPMR</div>
                     <div style="font-size:0.62rem;opacity:0.65;margin-top:0.2rem;">Liquid Metal · Gas Cooled · Heat Pipe</div>
                   </div>
                   <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                               border-radius:10px;padding:0.7rem 1.2rem;">
                     <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                                 opacity:0.6;margin-bottom:0.2rem;">Power Range</div>
                     <div style="font-weight:700;font-size:0.88rem;">1 – 20 MW<sub>t</sub></div>
                   </div>
                   <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                               border-radius:10px;padding:0.7rem 1.2rem;">
                     <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                                 opacity:0.6;margin-bottom:0.2rem;">Enrichment</div>
                     <div style="font-weight:700;font-size:0.88rem;">5 – 19.75%</div>
                   </div>
                   <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                               border-radius:10px;padding:0.7rem 1.2rem;">
                     <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                                 opacity:0.6;margin-bottom:0.2rem;">Outputs</div>
                     <div style="font-weight:700;font-size:0.88rem;">OCC · TCI · LCOE · LCOH · LCOF</div>
                     <div style="font-size:0.62rem;opacity:0.65;margin-top:0.2rem;">Capital Cost · Capital Investment · Cost of Energy · Cost of Heat · Cost of Fuel</div>
                   </div>
                   <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                               border-radius:10px;padding:0.7rem 1.2rem;">
                     <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                                 opacity:0.6;margin-bottom:0.2rem;">Analysis</div>
                     <div style="font-weight:700;font-size:0.88rem;">Parametric · Uncertainty · IRA Credits</div>
                   </div>
                 </div>
               </div>''',
            unsafe_allow_html=True,
        )

        st.markdown(
            '''<div style="background:#fffbeb;border:1.5px solid #f59e0b;border-radius:12px;
                           padding:1.2rem 1.5rem;margin-bottom:1rem;">
                 <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
                   <span style="font-size:1.2rem;">⚠️</span>
                   <span style="font-size:0.82rem;font-weight:800;color:#92400e;
                                text-transform:uppercase;letter-spacing:0.07em;">
                     Important Caveats — Please Read Before Use
                   </span>
                 </div>
                 <ul style="margin:0;padding-left:1.2rem;color:#78350f;font-size:0.84rem;line-height:1.75;">
                   <li><strong>Work in progress:</strong> This application is still under active development.
                       Features, results, and interfaces may change without notice.</li>
                   <li><strong>Pre-conceptual designs:</strong> The three reactor designs (LTMR, GCMR, HPMR)
                       are pre-conceptual. They have not been fully optimized and do not represent
                       final or licensed configurations.</li>
                   <li><strong>Incomplete information:</strong> Cost estimates were developed with incomplete
                       engineering and procurement data. Significant uncertainties exist across all
                       accounts, particularly for novel or first-of-a-kind components.</li>
                   <li><strong>Not for investment decisions:</strong> These estimates must <em>not</em> be
                       used as the basis for financial, investment, or procurement decisions. They are
                       intended solely for research, screening, and comparative analysis.</li>
                 </ul>
               </div>''',
            unsafe_allow_html=True,
        )

        st.info('Configure your reactor in the sidebar, then click **⚡ Run Cost Estimate** to begin.')

        st.markdown(
            """
            <div style='text-align: center; font-size: 0.9rem; color: gray; padding-top: 2rem; padding-bottom: 1rem;'>
                © 2025 Battelle Energy Alliance, LLC. MOUSE is released under the MIT License.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Run cost estimate ───────────────────────────────────────────────────
    with st.spinner('Running cost estimate…'):
        try:
            display_df, enriched_df, detailed_sorted_df, params = _run_estimate(
                reactor_type, power_mwt, enrichment,
                interest_rate / 100.0, discount_rate / 100.0, construction_duration, debt_to_equity,
                operation_mode, emergency_shutdowns, startup_duration, startup_duration_refueling,
                tax_credit_type, tax_credit_value, plant_lifetime,
                n_rings_per_assembly=n_rings_per_assembly,
                active_height=active_height,
                n_assembly_rings=n_assembly_rings,
                n_core_rings=n_core_rings,
            )
        except SubcriticalError as exc:
            st.error('### ⚠ Reactor is Subcritical')
            st.warning(str(exc))
            ca, cb, cc = st.columns(3)
            _info_card(ca, 'Fuel Lifetime', '0 days', accent='#dc2626', bg='#fef2f2', border='#fecaca')
            _info_card(cb, 'Thermal Power', f'{power_mwt} MW<sub>t</sub>',  accent='#9a3412', bg='#fff7ed', border='#fed7aa')
            _info_card(cc, 'Enrichment',    f'{enrichment*100:.2f}%', accent='#9a3412', bg='#fff7ed', border='#fed7aa')
            st.info('No cost estimate is available for a subcritical operating point. '
                    'Try reducing the power or increasing the enrichment.')

            st.markdown(
                """
                <div style='text-align: center; font-size: 0.9rem; color: gray; padding-top: 2rem; padding-bottom: 1rem;'>
                    © 2025 Battelle Energy Alliance, LLC. MOUSE is released under the MIT License.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.stop()
        except ShortLifetimeError as exc:
            # Extract the estimated lifetime in days from the message ("only N days")
            import re
            m = re.search(r'only (\d+) days', str(exc))
            est_lt_days = int(m.group(1)) if m else 0
            st.error('### ⚠ Fuel Lifetime Too Short')
            st.warning(str(exc))
            ca, cb, cc = st.columns(3)
            _info_card(ca, 'Fuel Lifetime', f'{est_lt_days} days',
                       accent='#dc2626', bg='#fef2f2', border='#fecaca')
            _info_card(cb, 'Thermal Power', f'{power_mwt} MW<sub>t</sub>',
                       accent='#9a3412', bg='#fff7ed', border='#fed7aa')
            _info_card(cc, 'Enrichment',    f'{enrichment*100:.2f}%',
                       accent='#9a3412', bg='#fff7ed', border='#fed7aa')
            st.info('No cost estimate is performed when the fuel lifetime is below 90 days. '
                    'Try increasing the diameter, the height, or the enrichment.')

            st.markdown(
                """
                <div style='text-align: center; font-size: 0.9rem; color: gray; padding-top: 2rem; padding-bottom: 1rem;'>
                    © 2025 Battelle Energy Alliance, LLC. MOUSE is released under the MIT License.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.stop()
        except Exception as exc:
            st.error(f'Cost estimation failed: {exc}')
            import traceback
            st.code(traceback.format_exc())

            st.markdown(
                """
                <div style='text-align: center; font-size: 0.9rem; color: gray; padding-top: 2rem; padding-bottom: 1rem;'>
                    © 2025 Battelle Energy Alliance, LLC. MOUSE is released under the MIT License.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.stop()

    # ── Extract key params ──────────────────────────────────────────────────
    fuel_lifetime   = params.get('Fuel Lifetime', float('nan'))
    power_mwe       = params.get('Power MWe', float('nan'))
    capacity_factor = params.get('Capacity Factor', float('nan'))

    _fl_str  = f'{fuel_lifetime / 365:.1f} yrs' if not math.isnan(float(fuel_lifetime)) else 'N/A'
    _fl_days = f'{int(fuel_lifetime):,} days'    if not math.isnan(float(fuel_lifetime)) else ''
    _cf_str  = f'{capacity_factor * 100:.1f}%'  if not math.isnan(float(capacity_factor)) else 'N/A'

    # ── Result hero banner ──────────────────────────────────────────────────
    _credit_badge = ''
    if tax_credit_type == 'PTC':
        _credit_badge = f'<span style="background:rgba(255,255,255,0.2);border:1px solid rgba(255,255,255,0.35);border-radius:999px;font-size:0.68rem;font-weight:700;padding:0.2rem 0.7rem;letter-spacing:0.06em;margin-left:0.6rem;">PTC ${tax_credit_value}/MWh</span>'
    elif tax_credit_type == 'ITC':
        _credit_badge = f'<span style="background:rgba(255,255,255,0.2);border:1px solid rgba(255,255,255,0.35);border-radius:999px;font-size:0.68rem;font-weight:700;padding:0.2rem 0.7rem;letter-spacing:0.06em;margin-left:0.6rem;">ITC {int(tax_credit_value*100)}%</span>'

    st.markdown(
        f'''<div style="background:linear-gradient(135deg,#0b1f3a 0%,#1B4F8C 55%,#1e6fa8 100%);
                       border-radius:16px;padding:1.8rem 2.2rem;color:white;
                       margin-bottom:1.5rem;position:relative;overflow:hidden;">
             <div style="position:absolute;right:2.5rem;top:50%;transform:translateY(-50%);
                         font-size:5.5rem;opacity:0.1;line-height:1;">⚛</div>
             <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;
                         text-transform:uppercase;opacity:0.55;margin-bottom:0.5rem;">
               Cost Estimate Result
             </div>
             <h2 style="font-size:1.65rem;font-weight:800;margin:0 0 0.9rem;color:white;line-height:1.2;">
               {reactor_label} {_credit_badge}
             </h2>
             <div style="display:flex;gap:2rem;flex-wrap:wrap;">
               <div>
                 <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.55;">Thermal Power</div>
                 <div style="font-weight:700;font-size:0.95rem;">{power_mwt} MW<sub>t</sub></div>
               </div>
               <div>
                 <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.55;">Enrichment</div>
                 <div style="font-weight:700;font-size:0.95rem;">{enrichment*100:.2f}%</div>
               </div>
               <div>
                 <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.55;">Interest Rate</div>
                 <div style="font-weight:700;font-size:0.95rem;">{interest_rate:.1f}%</div>
               </div>
               <div>
                 <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.55;">Discount Rate</div>
                 <div style="font-weight:700;font-size:0.95rem;">{discount_rate:.1f}%</div>
               </div>
               <div>
                 <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.55;">Operation Mode</div>
                 <div style="font-weight:700;font-size:0.95rem;">{operation_mode}</div>
               </div>
               <div>
                 <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.55;">Capacity Factor</div>
                 <div style="font-weight:700;font-size:0.95rem;">{_cf_str}</div>
               </div>
               <div>
                 <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.55;">All costs in</div>
                 <div style="font-weight:700;font-size:0.95rem;">{ESCALATION_YEAR} USD</div>
               </div>
             </div>
           </div>''',
        unsafe_allow_html=True,
    )

    # ── Collect all summary values ──────────────────────────────────────────
    if tax_credit_type == 'ITC':
        occ_account  = 'OCC (ITC-adjusted)'
        tci_account  = 'TCI (ITC-adjusted)'
        lcoe_account = 'LCOE (ITC-adjusted)'
    elif tax_credit_type == 'PTC':
        occ_account  = 'OCC'
        tci_account  = 'TCI'
        lcoe_account = 'LCOE with PTC'
    else:
        occ_account  = 'OCC'
        tci_account  = 'TCI'
        lcoe_account = 'LCOE'

    occ_f,  occ_f_std  = _get_mean_std(display_df, occ_account,  'FOAK')
    tci_f,  tci_f_std  = _get_mean_std(display_df, tci_account,  'FOAK')
    lcoe_f, lcoe_f_std = _get_mean_std(display_df, lcoe_account, 'FOAK')
    lcoh_f, lcoh_f_std = _get_mean_std(display_df, 'LCOH',       'FOAK')
    lcof_f, lcof_f_std = _get_lcof(enriched_df, 'FOAK')

    occ_n,  occ_n_std  = _get_mean_std(display_df, occ_account,  'NOAK')
    tci_n,  tci_n_std  = _get_mean_std(display_df, tci_account,  'NOAK')
    lcoe_n, lcoe_n_std = _get_mean_std(display_df, lcoe_account, 'NOAK')
    lcoh_n, lcoh_n_std = _get_mean_std(display_df, 'LCOH',       'NOAK')
    lcof_n, lcof_n_std = _get_lcof(enriched_df, 'NOAK')

    # ── Caveat reminder ─────────────────────────────────────────────────────
    st.markdown(
        '''<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;
                       padding:0.65rem 1rem;margin-bottom:1.2rem;
                       display:flex;align-items:flex-start;gap:0.6rem;">
             <span style="font-size:1rem;flex-shrink:0;margin-top:0.05rem;">⚠️</span>
             <span style="font-size:0.78rem;color:#92400e;line-height:1.55;">
               <strong>Caveats:</strong> This app is under active development. Reactor designs are
               pre-conceptual and not fully optimized. Cost estimates were produced with incomplete
               information and <strong>must not be used for investment or procurement decisions</strong>.
               Results are intended for research and comparative screening only.
             </span>
           </div>''',
        unsafe_allow_html=True,
    )

    tab_summary, tab_drivers, tab_table = st.tabs([
        '📊  Summary',
        '📈  Cost Drivers',
        '📋  Full Breakdown',
    ])

    # ═══════════════════════════════════════════════════════════════
    # TAB 1 — SUMMARY
    # ═══════════════════════════════════════════════════════════════
    with tab_summary:

        img_col, info_col = st.columns([1, 1], gap='large')

        with img_col:
            main_img, main_caption = _REACTOR_IMAGES[reactor_type]['main']

            # For LTMR, pick the per-N core cross-section that matches
            # the user's chosen diameter (Number of Rings per Assembly).
            if reactor_type == 'LTMR':
                _n = int(params.get('Number of Rings per Assembly', 12))
                _per_n_path = os.path.join(_ASSETS, f'LTMR_core_N{_n}.png')
                if os.path.exists(_per_n_path):
                    main_img = _per_n_path
                    main_caption = (
                        f"LTMR core cross-section for N = {_n} rings per assembly "
                        f"(active core radius {LTMR_N_TO_ACTIVE_RADIUS_CM[_n]:.1f} cm, "
                        f"active core diameter {LTMR_N_TO_DIAMETER_CM[_n]} cm). "
                        f"Hexagonal arrangement of TRIGA-type U-ZrH fuel pins and ZrH "
                        f"moderator pins cooled by NaK liquid metal, surrounded by a "
                        f"graphite radial reflector with 18 control drums."
                    )

            # For GCMR, pick the per-(N_A, N_C) core cross-section.
            if reactor_type == 'GCMR':
                _na = int(params.get('Assembly Rings', 6))
                _nc = int(params.get('Core Rings', 5))
                _per_pair_path = os.path.join(
                    _ASSETS, f'GCMR_core_NA{_na}_NC{_nc}.png'
                )
                if os.path.exists(_per_pair_path):
                    main_img = _per_pair_path
                    main_caption = (
                        f"GCMR core cross-section for (N_A={_na} assembly rings, "
                        f"N_C={_nc} core rings). Hexagonal fuel assemblies containing "
                        f"TRISO fuel compacts arranged in a honeycomb pattern, cooled "
                        f"by helium gas, with graphite reflector and control drums."
                    )

            # Cross-section on top, side-view directly below (LTMR/GCMR
            # only — both have geometry-driven height inputs). Both render
            # at the column width, so the side view's vertical extent
            # relative to the cross-section's diameter makes H/D visually
            # obvious.
            st.image(main_img, use_container_width=True)
            st.caption(main_caption)

            if reactor_type in ('LTMR', 'GCMR'):
                # The figure has built-in right padding to mirror the
                # cross-section's legend area, so the side-view rectangle
                # aligns horizontally with the cross-section circle above.
                # Save with bbox_inches=None (st.pyplot's default 'tight'
                # would crop the right-side empty space, defeating the
                # alignment).
                _diam = float(2.0 * params['Core Radius'])
                _h    = float(params['Active Height'])
                _ax_r = float(params.get('Axial Reflector Thickness', 0.0))
                _rad_r = float(params.get('Radial Reflector Thickness', 0.0))
                _fig = _make_side_view_figure(_diam, _h, _ax_r, _rad_r)
                _buf = io.BytesIO()
                _fig.savefig(_buf, format='png', bbox_inches=None,
                             facecolor='white', dpi=120)
                _buf.seek(0)
                st.image(_buf, use_container_width=True)
                plt.close(_fig)

            with st.expander('View detailed cross-sections'):
                for img_path, img_caption in _REACTOR_IMAGES[reactor_type]['details']:
                    # GCMR: replace the static fuel-assembly image with the
                    # per-N_A version that matches the user's selection.
                    if reactor_type == 'GCMR' and 'Fuel Assembly.png' in img_path:
                        _na = int(params.get('Assembly Rings', 6))
                        _per_na_path = os.path.join(
                            _ASSETS, f'GCMR_fuel_assembly_NA{_na}.png'
                        )
                        if os.path.exists(_per_na_path):
                            img_path = _per_na_path
                            img_caption = (
                                f"GCMR fuel assembly cross-section for N_A={_na} "
                                f"(assembly rings). TRISO fuel compacts, helium coolant "
                                f"channels, and ZrH moderator booster pins embedded in "
                                f"a graphite matrix."
                            )
                    st.image(img_path, use_container_width=True)
                    st.caption(img_caption)

        with info_col:
            ic1, ic2, ic3 = st.columns(3)
            _info_card(ic1, 'Electric Power Output', f'{power_mwe:.1f} MW<sub>e</sub>',
                       subtitle=f'Thermal input: {power_mwt} MW<sub>t</sub>')
            _info_card(ic2, 'Fuel Lifetime', _fl_str, subtitle=_fl_days)
            _info_card(ic3, 'Capacity Factor', _cf_str,
                       subtitle='Accounts for refueling & shutdowns')

            if reactor_type == 'LTMR':
                st.caption(
                    'Fuel Lifetime is estimated by KNN local regression on the LTMR '
                    'parametric study. Typical error is **5–15%** for cases close to '
                    'the trained design space, and may be larger for combinations '
                    'with non-trained ring counts (marked * in the diameter slider) '
                    'or near the criticality boundary.'
                )
            elif reactor_type == 'GCMR':
                st.caption(
                    'Fuel Lifetime is estimated by KNN local regression on the GCMR '
                    'parametric study. Typical error is **5–15%** for cases close to '
                    'the trained design space, and may be larger for combinations '
                    'with non-trained (Assembly Rings, Core Rings) pairs (marked * in '
                    'the diameter slider) or near the criticality boundary.'
                )

            st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

            # ── Reactivity vs Time (LTMR only) ──────────────────────────────
            _reactivity_swing_pct = None     # filled in for LTMR if curve available
            if reactor_type == 'LTMR':
                _times, _keffs = get_ltmr_keff_curve(
                    n_rings_per_assembly = params['Number of Rings per Assembly'],
                    active_height        = params['Active Height'],
                    enrichment           = params['Enrichment'],
                    power_mwt            = params['Power MWt'],
                    anchor_lifetime_days = params.get('Fuel Lifetime', None),
                )
                if _times.size >= 2:
                    _k_bol = float(_keffs[0])
                    if _k_bol > 1.0:
                        _reactivity_swing_pct = (_k_bol - 1.0) / _k_bol * 100.0
                if _times.size >= 2:
                    st.markdown(
                        '<div style="font-size:0.7rem;font-weight:700;color:#64748b;'
                        'text-transform:uppercase;letter-spacing:0.09em;'
                        'margin-bottom:0.6rem;">k_eff vs Time</div>',
                        unsafe_allow_html=True,
                    )
                    _kfig, _kax = plt.subplots(figsize=(5.5, 2.8))
                    # Show discrete interpolated points (markers) connected
                    # by straight segments — so the user can see we only
                    # have data at specific depletion timesteps, not a
                    # continuous curve.
                    _kax.plot(_times, _keffs, color='#1d4ed8', lw=1.5,
                              marker='o', markersize=5, markerfacecolor='#1d4ed8',
                              markeredgecolor='white', markeredgewidth=0.8)
                    _kax.fill_between(_times, _keffs, 1.0,
                                      where=(_keffs >= 1.0),
                                      color='#1d4ed8', alpha=0.10)
                    _kax.axhline(1.0, color='#7f1d1d', lw=1.0, ls='--')
                    _kax.set_xlabel('Time (days)', fontsize=9)
                    _kax.set_ylabel(r'$k_{\rm eff}$', fontsize=10)
                    _kax.tick_params(axis='both', labelsize=8)
                    _kax.set_xlim(left=0)
                    # Pad the y-axis a hair above the highest value for readability
                    _kax.set_ylim(0.98, max(1.005, _keffs.max() + 0.005))
                    _kax.grid(True, alpha=0.3)
                    for _spine in ('top', 'right'):
                        _kax.spines[_spine].set_visible(False)
                    _kfig.tight_layout()
                    st.pyplot(_kfig, use_container_width=True)
                    plt.close(_kfig)
                    st.caption(
                        'k_eff vs depletion time, interpolated from the 4 nearest '
                        'cases in the LTMR parametric study (distance-weighted '
                        'average of time and k_eff at each timestep). The time '
                        'axis is anchored so the k_eff = 1 crossing matches the '
                        'estimated fuel lifetime above; the subcritical tail is '
                        'omitted.'
                    )
                    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

            # ── Materials & Components (read directly from params) ──
            _materials_section(reactor_type, params)

            # ── Fuel Inventory (uses st.metric so we get the built-in help icon) ──
            _u235_g = float(params.get('Mass U235', 0.0))
            _u238_g = float(params.get('Mass U238', 0.0))
            _hm_g   = _u235_g + _u238_g
            _mwe    = float(params.get('Power MWe', 0.0)) or float('nan')
            _hm_kg          = _hm_g / 1.0e3                            # g → kg
            _hm_kg_per_mwe  = _hm_kg / _mwe                            # kg / MWe
            _fis_kg         = _u235_g / 1.0e3                          # g → kg
            _fis_kg_per_mwe = _fis_kg / _mwe                           # kg / MWe

            st.markdown(
                '<div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;'
                'letter-spacing:0.09em;margin-bottom:0.6rem;">Fuel Inventory</div>',
                unsafe_allow_html=True,
            )

            def _fuel_card(title, value_str, help_text,
                           accent='#0e7490', bg='#ecfeff', border='#a5f3fc'):
                # `title=` HTML attribute → native browser tooltip on hover.
                # Replace any " (which would close the attribute) with “ to be safe.
                _help_safe = help_text.replace('"', '“')
                st.markdown(
                    f'''<div style="background:{bg};border:1px solid {border};border-radius:14px;
                                    padding:1.0rem 1.25rem;box-shadow:0 2px 8px rgba(0,0,0,0.04);
                                    margin-bottom:0.6rem;">
                          <div style="font-size:0.68rem;font-weight:700;color:{accent};
                                      text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.35rem;
                                      display:flex;align-items:center;gap:0.4rem;">
                              <span>{title}</span>
                              <span title="{_help_safe}" style="cursor:help;color:#fff;
                                          background:{accent};border-radius:50%;
                                          width:16px;height:16px;display:inline-flex;
                                          align-items:center;justify-content:center;
                                          font-size:0.75rem;font-weight:800;line-height:1;">?</span>
                          </div>
                          <div style="font-size:1.05rem;font-weight:800;color:#111827;line-height:1.3;">
                              {value_str}
                          </div>
                        </div>''',
                    unsafe_allow_html=True,
                )

            _fuel_card(
                'Fuel loading',
                f'{_hm_kg:,.1f} kgHM  |  {_hm_kg_per_mwe:,.1f} kgHM/MWe',
                ('Total mass of heavy metal (uranium) in the core. '
                 'HM = Heavy Metal = Mass U-235 + Mass U-238. '
                 'kgHM/MWe normalises by net electric output — a specific fuel '
                 'inventory metric useful for comparing fuel-cycle requirements '
                 'across reactor types. Microreactors typically run '
                 '100-1,000 kgHM/MWe depending on technology.'),
            )
            _fuel_card(
                'Fissile loading',
                f'{_fis_kg:,.1f} kg  |  {_fis_kg_per_mwe:,.1f} kg/MWe',
                ('Mass of fissile material only (U-235 for LEU/HALEU fuel). '
                 'Drives enrichment cost (HALEU is expensive at ~$3k-$15k/kg) and '
                 'safeguards requirements. kg/MWe is one of the main metrics used '
                 'when comparing the economics of microreactors against larger '
                 'reactors — microreactor values are typically much higher than '
                 'commercial LWRs (~5 kg/MWe).'),
            )

            # Peaking factor + discharge burnup — only meaningful for
            # critical cases. Skip if HM mass or lifetime aren't available.
            _lifetime_days = float(params.get('Fuel Lifetime', 0.0))
            if _hm_kg > 0 and _lifetime_days > 0:
                _pf = 0.0
                if reactor_type == 'LTMR':
                    _pf = get_ltmr_peaking_factor(
                        n_rings_per_assembly = params['Number of Rings per Assembly'],
                        active_height        = params['Active Height'],
                        enrichment           = params['Enrichment'],
                        power_mwt            = params['Power MWt'],
                    )
                elif reactor_type == 'GCMR':
                    _pf = get_gcmr_peaking_factor(
                        assembly_rings = params['Assembly Rings'],
                        core_rings     = params['Core Rings'],
                        active_height  = params['Active Height'],
                        enrichment     = params['Enrichment'],
                        power_mwt      = params['Power MWt'],
                    )

                # Average discharge burnup: total energy / total HM mass.
                # MWt × days = MW·d, divided by kg → MWd/kg.
                _bu_avg = (float(params['Power MWt']) * _lifetime_days) / _hm_kg
                _bu_max = _bu_avg * _pf if _pf > 0 else 0.0

                _fuel_card(
                    'Peaking factor',
                    f'{_pf:,.2f}' if _pf > 0 else 'N/A',
                    ('Power Peaking Factor (PPF) — ratio of the maximum local '
                     'fission rate to the core-average fission rate. Drives the '
                     'difference between average and peak fuel pin temperatures, '
                     'and between average and peak burnup. Lower is better; '
                     'typical microreactor values are 1.5–3.0 depending on '
                     'reflector design and fuel arrangement. Interpolated from '
                     'the parametric study.'),
                    accent='#9333ea', bg='#faf5ff', border='#e9d5ff',
                )

                # Axial + total leakage (BOL, %).  Inside the trained
                # H range we use the KNN-interpolated value; outside we
                # fall back to a one-group migration-area physics
                # formula (which actually responds to the user's H).
                _ax_lk, _tot_lk, _lk_src = 0.0, 0.0, None
                _active_radius_cm = float(2.0 * params['Core Radius']) / 2.0
                # active_radius excludes the reflector — recompute from active diameter
                _active_radius_cm = (float(2.0 * params['Core Radius'])
                                     - 2.0 * float(params.get('Radial Reflector Thickness', 0.0))
                                     ) / 2.0
                _r_refl = float(params.get('Radial Reflector Thickness', 0.0))
                _z_refl = float(params.get('Axial Reflector Thickness', 0.0))
                if reactor_type == 'LTMR':
                    _ax_lk, _tot_lk, _lk_src = get_ltmr_leakage(
                        n_rings_per_assembly = params['Number of Rings per Assembly'],
                        active_height        = params['Active Height'],
                        enrichment           = params['Enrichment'],
                        power_mwt            = params['Power MWt'],
                        active_radius_cm     = _active_radius_cm,
                        radial_reflector_cm  = _r_refl,
                        axial_reflector_cm   = _z_refl,
                    )
                elif reactor_type == 'GCMR':
                    _ax_lk, _tot_lk, _lk_src = get_gcmr_leakage(
                        assembly_rings  = params['Assembly Rings'],
                        core_rings      = params['Core Rings'],
                        active_height   = params['Active Height'],
                        enrichment      = params['Enrichment'],
                        power_mwt       = params['Power MWt'],
                        active_radius_cm    = _active_radius_cm,
                        radial_reflector_cm = _r_refl,
                        axial_reflector_cm  = _z_refl,
                    )

                _src_note = (
                    'This value is INTERPOLATED from nearby cases in the '
                    'parametric study (KNN, K=4, distance-weighted average) — '
                    'the geometry sits inside the trained design space.'
                    if _lk_src == 'interpolated' else
                    'This value is COMPUTED from a one-group migration-area '
                    'physics formula because the requested geometry sits '
                    'outside the trained H range for this diameter — KNN '
                    'would just saturate at the training boundary. The '
                    'formula uses '
                    'B² = (2.405/R_eff)² + (π/H_eff)², '
                    'P_NL = 1/(1 + M²·B²), where R_eff and H_eff include '
                    'reflector savings (~0.6 × reflector thickness). M² '
                    'is calibrated against the parametric study data.'
                )

                if _ax_lk > 0:
                    _fuel_card(
                        'Axial leakage (BOL)',
                        f'{_ax_lk:,.2f} %',
                        ('Fraction of neutrons that leak out of the active core '
                         'through the top or bottom faces at beginning of life. '
                         'Driven primarily by Active Height (shorter cores leak '
                         'more axially) and the axial reflector thickness. The '
                         'reflector and the control drums are accounted for: '
                         'the parametric-study OpenMC runs include both, and '
                         'the physics fallback uses the auto-resolved '
                         'reflector thickness which already covers the drums. '
                         + _src_note),
                        accent='#0891b2', bg='#ecfeff', border='#a5f3fc',
                    )
                if _tot_lk > 0:
                    _fuel_card(
                        'Total leakage (BOL)',
                        f'{_tot_lk:,.2f} %',
                        ('Total fraction of neutrons that escape the active '
                         'core (axial + radial) at beginning of life. Driven '
                         'by core dimensions — both Active Height and active '
                         'radius. The reflector and the control drums are '
                         'accounted for the same way as above. Microreactors '
                         'typically have higher total leakage (~5–35 %) than '
                         'commercial LWRs (~3 %) because of their small size. '
                         + _src_note),
                        accent='#0891b2', bg='#ecfeff', border='#a5f3fc',
                    )
                _fuel_card(
                    'Discharge burnup (avg)',
                    f'{_bu_avg:,.1f} MWd/kgHM',
                    ('Average burnup of the fuel at end-of-life (when the reactor '
                     'first becomes subcritical). Computed as Power [MWt] × Fuel '
                     'Lifetime [days] / Heavy Metal mass [kg]. This is the '
                     'headline economic metric — higher discharge burnup means '
                     'more energy extracted per kg of fuel, lowering fuel cost '
                     'per MWh.'),
                    accent='#9333ea', bg='#faf5ff', border='#e9d5ff',
                )
                if _bu_max > 0:
                    _fuel_card(
                        'Discharge burnup (max)',
                        f'{_bu_max:,.1f} MWd/kgHM',
                        ('Peak burnup at end-of-life — the burnup of the most '
                         'heavily depleted region (= average × peaking factor). '
                         'This is the design-limiting value: cladding integrity, '
                         'fission gas release, and dimensional change all depend '
                         'on the peak. Commercial LWRs are licensed to ~62 '
                         'MWd/kgU peak. For TRISO-fuelled designs the peak burnup '
                         'limit is typically much higher.'),
                        accent='#9333ea', bg='#faf5ff', border='#e9d5ff',
                    )

                # Mining intensity uses MOUSE's existing 'Natural Uranium Mass'
                # (computed in fuel_calculations using T=0.25% tails and natural
                # U feed F=0.71%, the standard mass-balance formula).
                _nat_u_kg = float(params.get('Natural Uranium Mass', 0.0))
                _mwh_total = _mwe * _lifetime_days * 24.0
                if _nat_u_kg > 0 and _mwh_total > 0:
                    _mining = (_nat_u_kg * 1000.0) / _mwh_total   # kg→g, /MWh
                    _fuel_card(
                        'Mining intensity',
                        f'{_mining:,.1f} gU/MWh',
                        ('Mass of natural uranium that must be mined and milled '
                         'per MWh of electric energy produced. Computed from '
                         'MOUSE\'s natural-uranium consumption '
                         '(tails enrichment 0.25 %, feed enrichment 0.71 %), '
                         'then divided by the total lifetime electrical energy '
                         'produced. Typical values: commercial LWRs ~17–25 '
                         'gU/MWh, HALEU microreactors ~30–80, natural-U reactors '
                         '(CANDU) ~150–200. Lower means less front-end fuel-'
                         'cycle resource demand.'),
                        accent='#15803d', bg='#f0fdf4', border='#bbf7d0',
                    )

                if _reactivity_swing_pct is not None:
                    _fuel_card(
                        'Reactivity swing',
                        f'{_reactivity_swing_pct:,.1f} %Δk/k',
                        ('Total reactivity consumed by burnup over the fuel '
                         'cycle, in %Δk/k = (k_BOL − 1) / k_BOL × 100. k_BOL is '
                         'the interpolated beginning-of-life k_eff (fresh fuel, '
                         'before depletion). Drives control-drum sizing — drum '
                         'worth must exceed the swing to keep cold-clean k_eff '
                         '< 1 with all drums in. Typical: commercial LWRs ~10–'
                         '15 %, HALEU microreactors ~15–40 % (large because '
                         'long cycles + high enrichment).'),
                        accent='#9333ea', bg='#faf5ff', border='#e9d5ff',
                    )

                # Heat flux at the fuel-pin surface (MW/m²) — already in
                # params from calculate_heat_flux: Power / total pin
                # cylindrical surface area.
                _hflux = float(params.get('Heat Flux', 0.0))
                if _hflux > 0:
                    _fuel_card(
                        'Heat flux (avg)',
                        f'{_hflux * 100.0:,.1f} W/cm² ({_hflux:,.3f} MW/m²)',
                        ('Average heat flux at the outer surface of the fuel '
                         'pins = Power / (π × pin_diameter × H × pin_count). '
                         'Sets the convective heat-transfer requirement — the '
                         'coolant has to pull this much heat per unit area off '
                         'each pin. Typical microreactor values are 0.1–1 '
                         'MW/m² (10–100 W/cm²). Watch for departure-from-'
                         'nucleate-boiling (DNB) limits in liquid-cooled '
                         'designs and burnout limits in gas-cooled designs.'),
                        accent='#dc2626', bg='#fef2f2', border='#fecaca',
                    )

                # Separative Work Units (SWU): kg-SWU total and per MWh.
                _swu = float(params.get('SWU', 0.0))
                if _swu > 0 and _mwh_total > 0:
                    _swu_per_mwh = _swu * 1000.0 / _mwh_total       # kg → g, /MWh
                    _fuel_card(
                        'Enrichment SWU',
                        f'{_swu:,.0f} kg-SWU  |  {_swu_per_mwh:,.2f} g-SWU/MWh',
                        ('Separative Work Units consumed to produce the fuel '
                         'in this core — the standard metric for enrichment '
                         'effort. Computed in MOUSE\'s fuel_calculations using '
                         'the standard value-function method (tails 0.25 %, '
                         'feed 0.71 %). g-SWU/MWh normalises by total '
                         'electrical energy delivered. SWU directly drives '
                         'enrichment cost ($/kg-SWU varies, typically $50–'
                         '$200/kg-SWU). Higher enrichment products require '
                         'disproportionately more SWU per kg.'),
                        accent='#15803d', bg='#f0fdf4', border='#bbf7d0',
                    )

                # Onsite coolant inventory — power-scaled, reactor-specific.
                # LTMR: 1833 kg/MWt of NaK (Creys-Malville scaling).
                # GCMR: 3.3 kg/MWt of helium (UNT 919556 tables 17 & 18).
                # HPMR: 0 (heat pipes individually sealed; no bulk inventory).
                #
                # Display in tons with deliberately coarse rounding to signal
                # this is a rough estimate, not a precision figure:
                #   >= 1 ton  -> integer tons
                #   <  1 ton  -> 1 significant figure
                if reactor_type in ('LTMR', 'GCMR'):
                    _inv_kg = float(params.get('Onsite Coolant Inventory', 0.0))
                    if _inv_kg > 0:
                        _coolant_label = {'LTMR': 'NaK', 'GCMR': 'Helium'}[reactor_type]
                        _tons = _inv_kg / 1000.0
                        if _tons >= 1:
                            _val_str = f'{round(_tons):,d} ton'
                        else:
                            _val_str = f'{_tons:.1g} ton'
                        _inv_help = {
                            'LTMR': (
                                'Rough estimate of the on-site primary NaK '
                                'inventory (filled core + storage). Scales '
                                'linearly with thermal power at ~1833 kg/MWt, '
                                'derived from the Creys-Malville sodium plant '
                                '(5,500 t Na for a 3,000 MWt core). NaK does '
                                'not require periodic replacement — the '
                                'primary boundary is sealed and the coolant '
                                'is not consumed. Drives the coolant '
                                'procurement line in OCC. Value is rounded '
                                'coarsely to reflect estimate-level accuracy.'
                            ),
                            'GCMR': (
                                'Rough estimate of the on-site helium '
                                'inventory. Scales linearly with thermal '
                                'power at ~3.3 kg/MWt (UNT 919556 tables 17 '
                                '& 18). He has a steady ~10 %/year leakage '
                                'rate (NAS 12844), so one-tenth of this '
                                'inventory is replaced annually. Drives both '
                                'the OCC coolant line and the OPEX make-up '
                                'term. Value is rounded coarsely to reflect '
                                'estimate-level accuracy.'
                            ),
                        }[reactor_type]
                        _fuel_card(
                            'Coolant inventory',
                            f'{_val_str}  ({_coolant_label})',
                            _inv_help,
                            accent='#0e7490', bg='#ecfeff', border='#a5f3fc',
                        )

                # Coolant mass flow rate — only meaningful for LTMR (NaK).
                # GCMR uses helium gas; HPMR uses heat pipes (no flow).
                if reactor_type == 'LTMR':
                    _mdot = float(params.get('Coolant Mass Flow Rate', 0.0))
                    if _mdot > 0:
                        _fuel_card(
                            'Coolant mass flow rate',
                            f'{_mdot:,.1f} kg/s',
                            ('Primary-coolant mass flow rate computed from '
                             'm_dot = Power_MWt × 10⁶ / (ΔT × c_p) with the '
                             'LTMR\'s fixed ΔT = 90 °C across the core (Tin '
                             '430 → Tout 520 °C) and the heat capacity of '
                             'NaK eutectic. Sets the primary-pump and heat-'
                             'exchanger sizing. Liquid-metal microreactor '
                             'flow rates are typically 5–100 kg/s for the '
                             '1–20 MWt range; higher powers scale linearly.'),
                            accent='#0e7490', bg='#ecfeff', border='#a5f3fc',
                        )

                    _pump_kW = float(params.get('Primary Pump Mechanical Power', 0.0))
                    if _pump_kW > 0 and _mwe > 0:
                        _pump_pct = 100.0 * _pump_kW / (_mwe * 1000.0)
                        _fuel_card(
                            'Primary pump fraction',
                            f'{_pump_pct:,.2f} %  ({_pump_kW:,.0f} kW)',
                            ('Primary-loop pump mechanical power as a fraction '
                             'of the gross electric output. Computed from a '
                             'lumped pressure-drop model: P = m_dot × Δp / (ρ '
                             '× η), with default Δp = 250 kPa, ρ = 750 kg/m³ '
                             '(NaK at ~500 °C), η = 0.75. Typical liquid-metal '
                             'primary-loop pump fractions are 1–3 %; higher '
                             'values indicate an aggressively-loaded loop or '
                             'oversized core, lower values suggest natural-'
                             'circulation-dominated cooling.'),
                            accent='#0e7490', bg='#ecfeff', border='#a5f3fc',
                        )

                # Power density = Power_MWt / Active Core Volume.
                # Active core is a hex prism with apothem = active_radius
                # and height = Active Height. _active_radius_cm was already
                # computed earlier in this block (Core Radius − reflector).
                _vol_cm3 = (2.0 * _math.sqrt(3.0)
                            * (_active_radius_cm ** 2)
                            * float(params['Active Height']))
                _vol_m3 = _vol_cm3 * 1.0e-6
                if _vol_m3 > 0:
                    _pd = float(params['Power MWt']) / _vol_m3
                    _fuel_card(
                        'Power density',
                        f'{_pd:,.1f} MW/m³',
                        ('Thermal power per unit active core volume = '
                         'Power_MWt / (2√3 · R² · H), where R is the active '
                         'core hex apothem (no reflector) and H is the active '
                         'height. Tells you how aggressively the fuel is '
                         'loaded relative to the core size. Typical ranges: '
                         'eVinci/HPMR-class ~10–15 MW/m³, TRIGA/LTMR ~10–30, '
                         'metal-fuel microreactors (Oklo, BANR) ~50–100, '
                         'commercial LWRs ~100. Below ~5 MW/m³ means the core '
                         'is over-fuelled (long lifetime, low utilization); '
                         'above ~80 means the design is thermal-hydraulically '
                         'aggressive — also check the heat-flux card.'),
                        accent='#dc2626', bg='#fef2f2', border='#fecaca',
                    )

            st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

            # ─────────────────────────────────────────────────────────────
            # Transportability Considerations
            # ─────────────────────────────────────────────────────────────
            # Per-component dimensions (height, diameter) and dry mass for
            # the four nested envelopes that make up a microreactor module.
            # Per-mode badges compare the outermost (RVACS) envelope to
            # three transport-mode dimensional limits.  No total mass is
            # computed (per design choice — each component is shown alone).
            #
            # Rounding rules (match the coolant-inventory card):
            #   weights ≥ 1 ton  -> integer tons
            #   weights <  1 ton -> 1 significant figure
            #   dimensions       -> meters with 1 decimal
            st.markdown(
                '<div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;'
                'letter-spacing:0.09em;margin-bottom:0.6rem;">Transportability Considerations</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
                'padding:0.85rem 1.1rem;margin-bottom:0.9rem;font-size:0.82rem;line-height:1.45;color:#334155;">'
                'Transportability is one of the headline features of microreactors — the option to '
                'ship the reactor module in one piece on a truck, rail car, or sea container differentiates '
                'them from large NPPs. The values below check the current MOUSE configuration against the '
                'most relevant US transport-mode envelopes. Per-component masses are listed individually; '
                'no rolled-up total mass is reported.'
                '</div>',
                unsafe_allow_html=True,
            )

            # ── Build per-component rows (height_m, diameter_m, mass_kg) ──
            def _ton_str(kg):
                """Match the coolant-inventory rounding convention."""
                if kg is None:
                    return '—'
                try:
                    kg = float(kg)
                except (TypeError, ValueError):
                    return '—'
                if kg <= 0:
                    return '—'
                t = kg / 1000.0
                if t >= 1:
                    return f'{round(t):,d} ton'
                return f'{t:.1g} ton'

            def _m1(cm):
                if cm is None or cm <= 0:
                    return '—'
                return f'{cm/100.0:.1f} m'

            # MOUSE label disambiguation:
            #   - LTMR: 'Vessel' = reactor vessel, 'Guard Vessel' = guard vessel
            #   - GCMR: 'Vessel' = core barrel (internal),
            #           'Guard Vessel' = the actual RPV (He pressure boundary)
            #   - HPMR: 'Vessel' = reactor vessel, no guard vessel
            _vessel_height_cm = float(params.get('Vessel Height', 0.0))
            _bottom_depth_cm  = float(params.get('Vessel Bottom Depth', 0.0))
            _vessel_thk_cm    = float(params.get('Vessel Thickness', 0.0))
            _guard_thk_cm     = float(params.get('Guard Vessel Thickness', 0.0))
            _gap_v_g_cm       = float(params.get('Gap Between Vessel And Guard Vessel', 0.0))

            # Reactor (core + reflectors + drums).  All component masses
            # are in kg.  Note: 'Uranium Mass' is already in kg in MOUSE;
            # cladding mass is not separately tracked (small relative to
            # the other terms).
            _reactor_dia_cm  = 2.0 * float(params.get('Core Radius', 0.0))
            _reactor_h_cm    = (float(params.get('Active Height', 0.0))
                                + 2.0 * float(params.get('Axial Reflector Thickness', 0.0)))
            _reactor_mass_kg = (
                float(params.get('Uranium Mass', 0.0))
                + float(params.get('Moderator Mass', 0.0))
                + float(params.get('Moderator Booster Mass', 0.0))
                + float(params.get('Radial Reflector Mass', 0.0))
                + float(params.get('Axial Reflector Mass', 0.0))
                + float(params.get('Control Drums Mass', 0.0))
            )

            # Reactor vessel (the pressure boundary)
            if reactor_type == 'GCMR':
                # MOUSE 'Guard Vessel' is the RPV for GCMR
                _rv_outer_r_cm = (float(params.get('Guard Vessel Radius', 0.0))
                                  + _guard_thk_cm)
                _rv_height_cm  = (_vessel_height_cm
                                  + _bottom_depth_cm + _vessel_thk_cm + _gap_v_g_cm
                                  + _guard_thk_cm)
                _rv_mass_kg    = float(params.get('Guard Vessel Mass', 0.0))
            else:
                _rv_outer_r_cm = (float(params.get('Vessel Radius', 0.0)) + _vessel_thk_cm)
                _rv_height_cm  = _vessel_height_cm + _bottom_depth_cm
                _rv_mass_kg    = float(params.get('Vessel Mass', 0.0))
            _rv_dia_cm = 2.0 * _rv_outer_r_cm

            # Guard vessel — only for LTMR
            _has_guard = (reactor_type == 'LTMR'
                          and float(params.get('Guard Vessel Thickness', 0.0)) > 0)
            if _has_guard:
                _gv_outer_r_cm = (float(params.get('Guard Vessel Radius', 0.0)) + _guard_thk_cm)
                _gv_dia_cm     = 2.0 * _gv_outer_r_cm
                _gv_height_cm  = (_vessel_height_cm
                                  + _bottom_depth_cm + _vessel_thk_cm + _gap_v_g_cm
                                  + _guard_thk_cm)
                _gv_mass_kg    = float(params.get('Guard Vessel Mass', 0.0))
            else:
                _gv_dia_cm = _gv_height_cm = _gv_mass_kg = 0.0

            # RVACS (cooling vessel + intake vessel combined)
            _rvacs_outer_r_cm = float(params.get('Vessels Total Radius', 0.0))
            _rvacs_dia_cm     = 2.0 * _rvacs_outer_r_cm
            _rvacs_height_cm  = float(params.get('Vessels Total Height', 0.0))
            _rvacs_mass_kg    = (float(params.get('Cooling Vessel Mass', 0.0))
                                 + float(params.get('Intake Vessel Mass', 0.0)))

            # ── Render component table ──
            # Each row carries an always-visible small-font description
            # under the component name explaining what is included and
            # what is excluded.  The hover-only title= tooltip approach
            # was unreliable across browsers/themes, so the notes are
            # shown inline in the table itself.  Explicit colors
            # throughout so the table is readable regardless of the
            # Streamlit theme.
            _CELL = ('padding:0.55rem 0.8rem;color:#1e293b;'
                     'border-bottom:1px solid #e2e8f0;'
                     'vertical-align:top;')
            _CELL_C = _CELL + 'text-align:center;'
            _CELL_NAME = _CELL + 'font-weight:600;'
            _DESC = ('font-size:0.72rem;font-weight:400;color:#64748b;'
                     'line-height:1.4;margin-top:0.2rem;')

            _reactor_desc = (
                'Includes: U235 + U238, moderator '
                '(ZrH for LTMR; graphite ± ZrH booster for GCMR; '
                'monolith graphite for HPMR), radial + axial reflector, '
                'control drums. Excludes: fuel cladding (small, not '
                'tracked in MOUSE).'
                + (' <span style="color:#b45309;">HPMR — heat-pipe '
                   'steel cladding and Na working fluid not yet '
                   'modeled.</span>' if reactor_type == 'HPMR' else '')
            )
            _rv_desc_extra = (
                ' <span style="color:#0c4a6e;">For GCMR this maps to '
                'MOUSE\'s internal "Guard Vessel" field (the RPV).</span>'
                if reactor_type == 'GCMR' else ''
            )
            _rv_desc = (
                'Diameter = 2 × (vessel radius + thickness). '
                'Height = active core + axial reflector + lower '
                f'plenum + upper plenum + bottom dish '
                f'({_bottom_depth_cm:.1f} cm placeholder, flagged). '
                '<span style="color:#b45309;">Top closure dome '
                'not yet modeled.</span> Mass = vessel wall only.'
                + _rv_desc_extra
            )
            _gv_desc = (
                'Secondary containment shell around the reactor vessel '
                'for primary-coolant leak containment. Mass = '
                'guard-vessel wall only (no internals).'
            )
            _gv_na_desc = (
                'Intentionally omitted for this reactor type — He is '
                'inert (GCMR) and heat pipes are individually sealed '
                '(HPMR), so neither has a bulk primary coolant '
                'requiring secondary containment.'
            )
            _rvacs_desc = (
                'Cooling vessel + intake vessel treated as one shipping '
                'envelope. Diameter = 2 × intake vessel outer radius; '
                'height is the full external envelope. Mass = cooling-'
                'vessel wall + intake-vessel wall only (no air, no '
                'insulation, no support structure).'
            )

            _rows_html = []
            _rows_html.append(
                '<tr style="background:#ffffff;">'
                f'<td style="{_CELL_NAME}">Reactor (core + reflectors + drums)'
                f'<div style="{_DESC}">{_reactor_desc}</div></td>'
                f'<td style="{_CELL_C}">{_m1(_reactor_h_cm)}</td>'
                f'<td style="{_CELL_C}">{_m1(_reactor_dia_cm)}</td>'
                f'<td style="{_CELL_C}">{_ton_str(_reactor_mass_kg)}</td>'
                '</tr>'
            )
            _rows_html.append(
                '<tr style="background:#fafafa;">'
                f'<td style="{_CELL_NAME}">Reactor vessel'
                f'<div style="{_DESC}">{_rv_desc}</div></td>'
                f'<td style="{_CELL_C}">{_m1(_rv_height_cm)}</td>'
                f'<td style="{_CELL_C}">{_m1(_rv_dia_cm)}</td>'
                f'<td style="{_CELL_C}">{_ton_str(_rv_mass_kg)}</td>'
                '</tr>'
            )
            if _has_guard:
                _rows_html.append(
                    '<tr style="background:#ffffff;">'
                    f'<td style="{_CELL_NAME}">Guard vessel'
                    f'<div style="{_DESC}">{_gv_desc}</div></td>'
                    f'<td style="{_CELL_C}">{_m1(_gv_height_cm)}</td>'
                    f'<td style="{_CELL_C}">{_m1(_gv_dia_cm)}</td>'
                    f'<td style="{_CELL_C}">{_ton_str(_gv_mass_kg)}</td>'
                    '</tr>'
                )
            else:
                _rows_html.append(
                    '<tr style="background:#ffffff;">'
                    f'<td style="{_CELL_NAME};color:#94a3b8;">Guard vessel'
                    f'<div style="{_DESC}">{_gv_na_desc}</div></td>'
                    f'<td colspan="3" style="{_CELL_C};color:#94a3b8;">N/A — not used for this reactor type</td>'
                    '</tr>'
                )
            _rows_html.append(
                '<tr style="background:#fafafa;">'
                f'<td style="{_CELL_NAME}">RVACS (cooling + intake vessels)'
                f'<div style="{_DESC}">{_rvacs_desc}</div></td>'
                f'<td style="{_CELL_C}">{_m1(_rvacs_height_cm)}</td>'
                f'<td style="{_CELL_C}">{_m1(_rvacs_dia_cm)}</td>'
                f'<td style="{_CELL_C}">{_ton_str(_rvacs_mass_kg)}</td>'
                '</tr>'
            )

            _TH = ('padding:0.6rem 0.8rem;font-size:0.72rem;'
                   'text-transform:uppercase;letter-spacing:0.06em;'
                   'color:#475569;font-weight:700;')
            st.markdown(
                '<div style="margin-bottom:0.9rem;">'
                '<table style="width:100%;border-collapse:collapse;'
                'font-size:0.83rem;background:#ffffff;color:#1e293b;'
                'border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;">'
                '<thead style="background:#f1f5f9;">'
                '<tr>'
                f'<th style="{_TH};text-align:left;">Component</th>'
                f'<th style="{_TH};text-align:center;">Height</th>'
                f'<th style="{_TH};text-align:center;">Diameter</th>'
                f'<th style="{_TH};text-align:center;">Mass (dry, no coolant)</th>'
                '</tr></thead><tbody>'
                + ''.join(_rows_html) +
                '</tbody></table>'
                '</div>',
                unsafe_allow_html=True,
            )

            # ── Always-visible notes panel ──
            # Hover tooltips on the (?) icons are easy to miss; spell out
            # what each component includes/excludes and the global
            # assumptions below the table.
            _hpmr_note = (
                '<li><strong>HPMR-specific:</strong> heat-pipe steel '
                'cladding and the Na working fluid are <em>not</em> '
                'currently modeled in MOUSE — flagged for future addition.</li>'
                if reactor_type == 'HPMR' else ''
            )
            _gcmr_note = (
                '<li><strong>GCMR labeling note:</strong> what is shown '
                'as "Reactor vessel" here maps to MOUSE\'s internal '
                '<em>Guard Vessel</em> field, because for the GCMR the '
                'outer pressure shell is the RPV, not the inner core barrel.</li>'
                if reactor_type == 'GCMR' else ''
            )
            _gv_note = (
                '<li><strong>Guard vessel:</strong> wall thickness and '
                'gap; mass = wall only (no internals).</li>'
                if _has_guard else
                '<li><strong>Guard vessel — N/A:</strong> intentionally '
                'omitted for this reactor type (helium is inert for GCMR; '
                'each heat pipe is individually sealed for HPMR — neither '
                'has a bulk primary coolant requiring secondary containment).</li>'
            )
            st.markdown(
                '<div style="background:#f0f9ff;border:1px solid #bae6fd;'
                'border-radius:10px;padding:0.85rem 1.1rem;margin-bottom:0.9rem;'
                'font-size:0.79rem;line-height:1.55;color:#0c4a6e;">'
                '<div style="font-weight:700;font-size:0.72rem;'
                'text-transform:uppercase;letter-spacing:0.06em;'
                'color:#0369a1;margin-bottom:0.45rem;">Notes &amp; assumptions</div>'
                '<ul style="margin:0;padding-left:1.2rem;color:#0c4a6e;">'
                '<li><strong>Reactor (core + reflectors + drums):</strong> '
                'mass sums uranium (U235 + U238) + moderator (ZrH for LTMR; '
                'graphite + ZrH booster for GCMR; monolith graphite for HPMR) '
                '+ radial reflector + axial reflector + control drums. '
                'Fuel-pin cladding is not separately tracked in MOUSE '
                '(small relative to the other terms).</li>'
                + _hpmr_note +
                '<li><strong>Reactor vessel:</strong> diameter is '
                '2 × (vessel radius + vessel thickness); height includes '
                f'active core, axial reflector, lower plenum, upper plenum, '
                f'and bottom dish (currently using the placeholder Vessel '
                f'Bottom Depth = {_bottom_depth_cm:.1f} cm — flagged for '
                'review). Top closure dome is <em>not</em> currently '
                'modeled in MOUSE — flagged for future addition. Mass is '
                'the vessel wall only, no internals.</li>'
                + _gcmr_note
                + _gv_note +
                '<li><strong>RVACS (cooling + intake vessels):</strong> '
                'treated as one shipping envelope. Diameter = 2 × intake '
                'vessel outer radius; height is the full external envelope '
                'including bottom dish. Mass = cooling-vessel wall + '
                'intake-vessel wall only (no air, no insulation, no '
                'support structure).</li>'
                '<li><strong>Shielding excluded:</strong> in-vessel '
                'shielding (B<sub>4</sub>C) and out-of-vessel shielding '
                '(WEP / concrete biological shield) are <em>not</em> '
                'considered in this section. Shielding adds significant '
                'mass and outer dimension to the as-shipped or as-installed '
                'module, depending on whether it ships with the reactor or '
                'is built on site.</li>'
                '<li><strong>Coolant excluded:</strong> primary coolant '
                'inventory (NaK for LTMR, He for GCMR, heat-pipe Na for '
                'HPMR) is <em>not</em> included in the dry-mass column.</li>'
                '<li><strong>Other excluded items:</strong> support skirt, '
                'lifting lugs, transport frame, control-rod drives, and '
                'any auxiliary piping outside the four nested vessels.</li>'
                '</ul>'
                '</div>',
                unsafe_allow_html=True,
            )

            # ── Transport-mode dimensional limits + badges ──
            # Limits & references
            _modes = [
                {
                    'name': 'US road, no permit',
                    'width_m':  2.59,   # 8.5 ft
                    'height_m': 4.11,   # 13.5 ft (state-varying)
                    'length_m': None,   # length rarely binding for a vessel module
                    'weight_t': 36.3,   # 80,000 lb GVW
                    'cite_html': (
                        'Federal Bridge Formula — '
                        '<a href="https://www.law.cornell.edu/uscode/text/23/127" '
                        'target="_blank" style="color:#2563eb;">23 U.S.C. § 127</a>; '
                        'FHWA "Federal Size Regulations for Commercial Motor Vehicles". '
                        'State-by-state variations may permit larger envelopes on '
                        'non-Interstate routes.'
                    ),
                },
                {
                    'name': 'US rail, AAR Plate F',
                    'width_m':  5.18,
                    'height_m': 5.18,
                    'length_m': None,
                    'weight_t': 130.0,  # 286,000 lb gross car weight
                    'cite_html': (
                        'AAR Manual of Standards & Recommended Practices §C-II '
                        '"Plate Drawings"; AAR Open Top Loading Rules. '
                        'Plate F is the standard high-clearance envelope for '
                        'oversized industrial cargo.'
                    ),
                },
                {
                    'name': '40 ft ISO HC sea container',
                    'width_m':  2.352,  # internal
                    'height_m': 2.700,  # internal
                    'length_m': 12.032, # internal
                    'weight_t': 30.5,
                    'cite_html': (
                        'ISO 668:2020 "Series 1 freight containers — Classification, '
                        'dimensions and ratings"; ISO 1496-1:2013 "Specification '
                        'and testing — Part 1: General-cargo containers".'
                    ),
                },
            ]

            # Compare the outermost RVACS envelope to each mode's
            # dimensional limits AND weight limit.  Weight comparison
            # uses the sum of the four component masses (computed inline
            # for badge logic only — not displayed as a "total" row in
            # the table per design choice).
            _rvacs_dia_m = _rvacs_dia_cm / 100.0
            _rvacs_h_m   = _rvacs_height_cm / 100.0
            _badge_total_kg = (_reactor_mass_kg + _rv_mass_kg
                               + _gv_mass_kg + _rvacs_mass_kg)
            _badge_total_t  = _badge_total_kg / 1000.0

            _mode_cards_html = []
            for _mode in _modes:
                _w_ok = _rvacs_dia_m <= _mode['width_m']
                _h_ok = _rvacs_h_m   <= _mode['height_m']
                _len_ok  = (_mode['length_m'] is None) or (_rvacs_h_m <= _mode['length_m'])
                _wt_ok   = _badge_total_t <= _mode['weight_t']
                _fits    = _w_ok and _h_ok and _len_ok and _wt_ok
                _badge_color = ('#15803d', '#dcfce7', '#bbf7d0') if _fits else ('#b91c1c', '#fee2e2', '#fecaca')
                _badge_text  = '✓ fits envelope' if _fits else '✗ exceeds envelope'
                _len_str = (f' &nbsp;|&nbsp; length ≤ {_mode["length_m"]:.2f} m'
                            if _mode['length_m'] is not None else '')
                _mode_cards_html.append(
                    '<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;'
                    'padding:0.85rem 1.1rem;margin-bottom:0.6rem;color:#1e293b;">'
                    '<div style="display:flex;justify-content:space-between;align-items:center;'
                    'margin-bottom:0.35rem;">'
                    f'<div style="font-weight:700;font-size:0.86rem;color:#1e293b;">{_mode["name"]}</div>'
                    f'<div style="background:{_badge_color[1]};border:1px solid {_badge_color[2]};'
                    f'color:{_badge_color[0]};font-size:0.72rem;font-weight:700;'
                    f'padding:0.15rem 0.6rem;border-radius:6px;">{_badge_text}</div>'
                    '</div>'
                    f'<div style="font-size:0.78rem;color:#475569;margin-bottom:0.3rem;">'
                    f'width ≤ {_mode["width_m"]:.2f} m &nbsp;|&nbsp; '
                    f'height ≤ {_mode["height_m"]:.2f} m &nbsp;|&nbsp; '
                    f'weight ≤ {_mode["weight_t"]:.1f} ton{_len_str}'
                    f'</div>'
                    f'<div style="font-size:0.72rem;color:#64748b;line-height:1.4;">{_mode["cite_html"]}</div>'
                    '</div>'
                )

            st.markdown(''.join(_mode_cards_html), unsafe_allow_html=True)

            # Footnote — what each badge does and doesn't check
            st.markdown(
                '<div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;'
                'padding:0.65rem 1rem;margin-bottom:1rem;font-size:0.74rem;line-height:1.45;color:#7c2d12;">'
                '<strong>Note:</strong> Each badge compares the outermost RVACS envelope '
                '(diameter, height) and the sum of all component masses against the mode\'s '
                'limits. Per-component dimensions and masses are shown above; component-level '
                'help icons (?) explain what is and isn\'t included for each row. Top closure '
                'dome and bottom dish are not currently modeled in '
                'MOUSE, so the height values are slight underestimates.'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;'
                'letter-spacing:0.09em;margin-bottom:0.6rem;">Capital Costs</div>',
                unsafe_allow_html=True,
            )
            cc1, cc2 = st.columns(2)
            _kpi_card(cc1, 'Overnight Capital Cost (OCC)',
                      _fmt_cost(occ_f, occ_f_std), _fmt_cost(occ_n, occ_n_std),
                      color=_CARD_COLORS['occ'])
            _kpi_card(cc2, 'Total Capital Investment (TCI)',
                      _fmt_cost(tci_f, tci_f_std), _fmt_cost(tci_n, tci_n_std),
                      color=_CARD_COLORS['tci'])

            st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

            st.markdown(
                '<div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;'
                'letter-spacing:0.09em;margin-bottom:0.6rem;">Levelized Costs</div>',
                unsafe_allow_html=True,
            )
            lc1, lc2, lc3 = st.columns(3)
            _kpi_card(lc1, 'LCOE ($/MW<sub>e</sub>h)',
                      _fmt_lcoe(lcoe_f, lcoe_f_std), _fmt_lcoe(lcoe_n, lcoe_n_std),
                      color=_CARD_COLORS['lcoe'])
            _kpi_card(lc2, 'LCOH ($/MW<sub>t</sub>h)',
                      _fmt_lcoh(lcoh_f, lcoh_f_std), _fmt_lcoh(lcoh_n, lcoh_n_std),
                      color=_CARD_COLORS['lcoh'])
            _kpi_card(lc3, 'LCOF — Fuel Only ($/MW<sub>e</sub>h)',
                      _fmt_lcoe(lcof_f, lcof_f_std), _fmt_lcoe(lcof_n, lcof_n_std),
                      color=_CARD_COLORS['lcof'])

        st.markdown(
            '<div style="margin-top:1.2rem;padding:0.7rem 1rem;background:#f8fafc;'
            'border-radius:8px;border:1px solid #e2e8f0;font-size:0.78rem;color:#64748b;">'
            '<span style="background:#fff3ed;color:#c84b1e;font-size:0.62rem;font-weight:800;'
            'padding:0.1rem 0.4rem;border-radius:4px;margin-right:0.4rem;">FOAK</span>'
            'First-of-a-Kind — includes learning curve and contingency premiums for an initial deployment.&emsp;'
            '<span style="background:#eff6ff;color:#1d4ed8;font-size:0.62rem;font-weight:800;'
            'padding:0.1rem 0.4rem;border-radius:4px;margin-right:0.4rem;">NOAK</span>'
            'Nth-of-a-Kind — reflects mature, serial production cost reductions at scale.'
            '</div>',
            unsafe_allow_html=True,
        )

        # ─────────────────────────────────────────────────────────
        # Costs in perspective: NOAK LCOE vs market benchmarks
        # ─────────────────────────────────────────────────────────
        st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;'
            'letter-spacing:0.09em;margin-bottom:0.6rem;">Costs in Perspective</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
            'padding:0.85rem 1.1rem;margin-bottom:0.9rem;font-size:0.82rem;line-height:1.45;color:#334155;">'
            'The curve is anchored at three deployment scales: N=1 (FOAK from the headline '
            'card), N=10 (one extra cost-engine call), and N=user-set NOAK Unit Number '
            '(default 100, NOAK from the headline card). The shaded band reflects the '
            'one-sigma uncertainty around each anchor and is connected piecewise between '
            'them. Overlaid against seven US-relevant electricity-market price ranges to '
            'indicate where the reactor would be cost-competitive at each scale.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Use the headline FOAK (N=1) and NOAK (N=user_setting) values
        # as anchors, and add ONE extra cost-engine call at N=10 to
        # give an intermediate point.  Same extraction path as the
        # headline cards (bottom_up_cost_estimate ->
        # cost_drivers_estimate -> transform_dataframe ->
        # _get_mean_std), so values are consistent by construction.
        _N_user = int(round(float(params.get('NOAK Unit Number', 100))))
        if _N_user < 2:
            _N_user = 100

        # Compute the intermediate anchor at N=10 (one cost-engine
        # call, cached on inputs).
        _N_mid = 10
        _mid_diag_df = None
        _mid_diag_params = {}
        with st.spinner('Computing intermediate LCOE anchor (N=10)…'):
            try:
                # CRITICAL: interest_rate / 100 and discount_rate / 100
                # to convert UI percent to fraction.  Headline call
                # does the same conversion at line ~1537; the N=10
                # helper has to mirror it or the cost engine treats
                # 7 as 700% interest and explodes the IDC / TCI.
                _mid_m, _mid_s, _mid_diag_df, _mid_diag_params = _lcoe_at_noak_unit(
                    reactor_type, power_mwt, enrichment,
                    interest_rate / 100.0, discount_rate / 100.0,
                    construction_duration,
                    debt_to_equity, operation_mode, emergency_shutdowns,
                    startup_duration, startup_duration_refueling,
                    tax_credit_type, tax_credit_value, plant_lifetime,
                    n_rings_per_assembly=n_rings_per_assembly,
                    active_height=active_height,
                    n_assembly_rings=n_assembly_rings,
                    n_core_rings=n_core_rings,
                    noak_unit_number=_N_mid,
                )
            except Exception as _e:
                _mid_m, _mid_s = float('nan'), 0.0
                st.warning(f'Could not compute N=10 anchor: {_e}')

        # Show param snapshot + per-account diagnostic for the N=10
        # call so we can see exactly what the cost engine saw and
        # which account drives the runaway value.
        if _mid_diag_params:
            _params_lines = '\n'.join(
                f'  {k:32s} = {v}' for k, v in _mid_diag_params.items()
            )
            st.markdown(
                '<div style="background:#fef3c7;border:1px solid #fcd34d;'
                'border-radius:8px;padding:0.6rem 0.9rem;margin-bottom:0.7rem;'
                'font-family:ui-monospace,Menlo,monospace;font-size:0.75rem;'
                'color:#78350f;white-space:pre;">'
                f'<strong style="font-family:inherit;">Params used in N={_N_mid} call:</strong>\n'
                + _params_lines +
                '</div>',
                unsafe_allow_html=True,
            )
        if _mid_diag_df is not None and not _mid_diag_df.empty:
            with st.expander(
                f'Per-account costs at NOAK Unit Number = {_N_mid} '
                f'(diagnostic — close once verified)',
                expanded=False,
            ):
                st.dataframe(_mid_diag_df, use_container_width=True)

        try:
            _foak_m = float(lcoe_f)
            _noak_m = float(lcoe_n)
        except (TypeError, ValueError):
            _foak_m = _noak_m = float('nan')
        try:
            _foak_s = float(lcoe_f_std) if lcoe_f_std == lcoe_f_std else 0.0
            _noak_s = float(lcoe_n_std) if lcoe_n_std == lcoe_n_std else 0.0
        except (TypeError, ValueError):
            _foak_s = _noak_s = 0.0

        # Build the anchor list — include N=10 only if it's between
        # N=1 and N_user and the value came back valid.
        if (_mid_m == _mid_m and _N_mid > 1 and _N_mid < _N_user):
            _units = [1, _N_mid, _N_user]
            _means = [_foak_m, _mid_m, _noak_m]
            _stds  = [_foak_s, _mid_s, _noak_s]
        else:
            _units = [1, _N_user]
            _means = [_foak_m, _noak_m]
            _stds  = [_foak_s, _noak_s]

        # ── Diagnostic dump (always visible) so we can debug ──────────
        # Print the raw sweep results in a fixed-width panel.  This
        # lets us see exactly what the cost engine returned for each
        # NOAK Unit Number, even if the plot rendering fails for any
        # reason.
        _diag_lines = []
        for _i, _N in enumerate(_units or []):
            _m = _means[_i] if _i < len(_means) else float('nan')
            _s = _stds[_i]  if _i < len(_stds)  else float('nan')
            _diag_lines.append(f'  N = {int(_N):>3d}  mean = ${_m:.1f}  std = ${_s:.1f}')
        if _diag_lines:
            st.markdown(
                '<div style="background:#fff8db;border:1px solid #fde68a;'
                'border-radius:8px;padding:0.6rem 0.9rem;margin-bottom:0.7rem;'
                'font-family:ui-monospace,Menlo,monospace;font-size:0.78rem;'
                'color:#78350f;white-space:pre;">'
                '<strong style="font-family:inherit;">Sweep diagnostic (NOAK LCOE per unit count, $/MWh):</strong>\n'
                + '\n'.join(_diag_lines) +
                '</div>',
                unsafe_allow_html=True,
            )

        # NaN-tolerant gating: drop points where the LCOE came back
        # NaN, plot whatever valid points remain.  Scatter markers +
        # mean line are always drawn (they don't need the spline).
        _u_arr = np.array(_units, dtype=float) if _units else np.array([])
        _m_arr = np.array(_means, dtype=float) if _means else np.array([])
        _s_arr = np.array(_stds,  dtype=float) if _stds  else np.array([])
        if _m_arr.size:
            _valid = ~np.isnan(_m_arr)
            _u_arr = _u_arr[_valid]
            _m_arr = _m_arr[_valid]
            _s_arr = _s_arr[_valid]
            _s_arr = np.nan_to_num(_s_arr, nan=0.0)

        if _u_arr.size >= 2:
            from matplotlib.patches import Patch as _Patch

            # Reactor-type-specific palette (matches the existing
            # webapp accent colors for each reactor).
            _palette = {
                'LTMR': ('#4472C4', '#2E5090'),  # blue
                'GCMR': ('#FF6B6B', '#D94444'),  # red
                'HPMR': ('#7E57C2', '#4527A0'),  # purple
            }
            _fill_color, _edge_color = _palette.get(reactor_type, _palette['LTMR'])
            _legend_label = {
                'LTMR': 'Liquid-Metal Microreactor',
                'GCMR': 'Gas-Cooled Microreactor',
                'HPMR': 'Heat Pipe Microreactor',
            }.get(reactor_type, reactor_type)

            _fig, _ax = plt.subplots(figsize=(11, 5.8))

            # Plain linear interpolation between the anchor points —
            # no spline, no overshoot.  np.interp gives straight line
            # segments between consecutive anchors, which is the
            # honest representation when we only have 2-3 anchors.
            try:
                _x_smooth = np.linspace(_u_arr.min(), _u_arr.max(), 300)
                _m_smooth = np.interp(_x_smooth, _u_arr, _m_arr)
                _s_smooth = np.interp(_x_smooth, _u_arr, _s_arr)
            except Exception:
                _x_smooth = _u_arr
                _m_smooth = _m_arr
                _s_smooth = _s_arr

            _ax.fill_between(_x_smooth,
                             _m_smooth - _s_smooth,
                             _m_smooth + _s_smooth,
                             color=_fill_color, alpha=0.45, label=_legend_label,
                             edgecolor=_edge_color, linewidth=1.5, zorder=2)
            # Mean line + scatter markers at the actual computed
            # sweep points so the LCOE curve is always visible even if
            # the spline interpolation produces something degenerate.
            _ax.plot(_x_smooth, _m_smooth, color=_edge_color,
                     linewidth=2.0, linestyle='-', zorder=3)
            _ax.scatter(_u_arr, _m_arr, s=42, color=_edge_color,
                        edgecolor='white', linewidth=1.2, zorder=4)

            # Market benchmark arrows (same coords as the reference figure)
            _markets = {
                'Remote communities':           {'x':  2, 'y_start': 400, 'y_end': 290, 'color': '#C00000', 'arrow_only_down': True},
                'Defense':                      {'x':  8, 'y_start': 316, 'y_end': 296, 'color': '#FFC000', 'arrow_only_down': False},
                'Island & Mining':              {'x': 30, 'y_start': 380, 'y_end': 190, 'color': '#ED7D31', 'arrow_only_down': False},
                'Alaska railbelt electricity':  {'x': 48, 'y_start': 313, 'y_end': 182, 'color': '#7030A0', 'arrow_only_down': False},
                'Alaska railbelt generation':   {'x': 60, 'y_start': 166, 'y_end':  62, 'color': '#A6A6A6', 'arrow_only_down': False},
                'U.S. grid electricity':        {'x': 75, 'y_start': 270, 'y_end':  79, 'color': '#92D050', 'arrow_only_down': False},
                'U.S. grid generation':         {'x': 88, 'y_start':  55, 'y_end':  29, 'color': '#00B050', 'arrow_only_down': False},
            }
            _bar_widths = {
                'Remote communities':           4,
                'Defense':                      4,
                'Island & Mining':             12,
                'Alaska railbelt electricity':  8,
                'Alaska railbelt generation':   8,
                'U.S. grid electricity':       12,
                'U.S. grid generation':        10,
            }
            _label_offsets = {
                'Remote communities':         +8,
                'Defense':                   +25,
                'Alaska railbelt generation': -15,
                'U.S. grid generation':      +20,
            }
            for _name, _d in _markets.items():
                _x  = _d['x']
                _ys = min(_d['y_start'], 400)
                _ye = _d['y_end']
                _c  = _d['color']
                _down = _d['arrow_only_down']
                _w  = _bar_widths[_name]
                _xa, _xb = _x, _x + _w
                _xm = _x + _w / 2

                if not _down:
                    _ax.plot([_xa, _xb], [_ys, _ys], color=_c, linewidth=4,
                             solid_capstyle='round', zorder=5)
                _ax.plot([_xa, _xb], [_ye, _ye], color=_c, linewidth=4,
                         solid_capstyle='round', zorder=5)
                _astyle = '->' if _down else '<->'
                if _down:
                    _ax.plot([_xm, _xm], [_ys, _ye], color=_c, linewidth=2.8,
                             solid_capstyle='round', zorder=5)
                _ax.annotate('', xy=(_xm, _ye), xytext=(_xm, _ys),
                             arrowprops=dict(arrowstyle=_astyle, color=_c,
                                             lw=2.8, mutation_scale=16), zorder=5)

                if _name == 'Alaska railbelt generation':
                    _ly = _ye + _label_offsets[_name]
                else:
                    _ly = _ys + _label_offsets.get(_name, +15)
                _ax.text(_xm, _ly, _name, fontsize=8.5, ha='center', color=_c,
                         fontweight='bold',
                         bbox=dict(facecolor='white', edgecolor=_c,
                                   boxstyle='round,pad=0.25',
                                   linewidth=1.0, alpha=0.9),
                         zorder=6)

            _ax.set_xlim(0, 100)
            # Y-axis sizing — use the 90th-percentile of (mean + std)
            # rather than max, so a single outlier point doesn't blow
            # up the y-range and squash the rest of the data.  Hard
            # cap at $800 so even outliers stay readable.
            if _m_arr.size:
                _band_vals = _m_arr + _s_arr
                _band_p90 = float(np.nanpercentile(_band_vals, 90))
            else:
                _band_p90 = 0.0
            _ymax = max(410.0, _band_p90 * 1.15)
            _ymax = min(_ymax, 800.0)
            _ax.set_ylim(0, _ymax)
            _ax.set_xlabel('Number of Units Deployed', fontsize=12, fontweight='bold')
            _ax.set_ylabel('LCOE ($/MWh)', fontsize=12, fontweight='bold')
            _ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            from matplotlib.ticker import MaxNLocator
            _ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
            _ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            _ax.set_axisbelow(True)
            _ax.set_facecolor('white')
            _fig.patch.set_facecolor('white')

            _legend_patch = _Patch(facecolor=_fill_color, edgecolor=_edge_color,
                                   alpha=0.6, label=_legend_label)
            _ax.legend(handles=[_legend_patch], fontsize=10, loc='upper right',
                       framealpha=0.9, edgecolor='grey')

            _fig.tight_layout()
            st.pyplot(_fig)
            plt.close(_fig)

            # Prominent caption with the raw sweep points so the user
            # can verify the curve at a glance.  Shown in a soft-blue
            # panel with monospace font for the numbers.
            _pts_str = ' · '.join(
                f'N={int(u)}: ${m:.0f}±{s:.0f}'
                for u, m, s in zip(_u_arr, _m_arr, _s_arr)
            )
            st.markdown(
                f'<div style="background:#f0f9ff;border:1px solid #bae6fd;'
                f'border-radius:8px;padding:0.55rem 0.9rem;margin-top:0.2rem;'
                f'margin-bottom:0.9rem;font-size:0.78rem;color:#0c4a6e;">'
                f'<strong>NOAK LCOE per sweep point ($/MWh):</strong> '
                f'<span style="font-family:ui-monospace,Menlo,monospace;'
                f'color:#0f172a;">{_pts_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Market definitions panel (matches the user-provided spec)
            st.markdown(
                '<div style="background:#f0f9ff;border:1px solid #bae6fd;'
                'border-radius:10px;padding:0.85rem 1.1rem;margin-bottom:0.9rem;'
                'font-size:0.78rem;line-height:1.55;color:#0c4a6e;">'
                '<div style="font-weight:700;font-size:0.72rem;'
                'text-transform:uppercase;letter-spacing:0.06em;'
                'color:#0369a1;margin-bottom:0.45rem;">Market definitions</div>'
                '<ul style="margin:0;padding-left:1.2rem;">'
                '<li><strong>U.S. Grid Generation:</strong> regional average wholesale price; '
                'excludes transmission, distribution, and customer charges.</li>'
                '<li><strong>U.S. Grid Electricity:</strong> state-level average retail '
                'electricity price, all sectors; excludes Alaska and Hawaii.</li>'
                '<li><strong>Alaska Railbelt Generation:</strong> wholesale generation cost; '
                'excludes transmission, distribution, and customer charges.</li>'
                '<li><strong>Alaska Railbelt Electricity:</strong> retail price including '
                'generation, transmission, distribution, and adjustments.</li>'
                '<li><strong>Island &amp; Mining:</strong> all-in diesel- or LNG-based '
                'electricity cost, including fuel delivery.</li>'
                '<li><strong>Defense:</strong> remote base electricity cost plus a premium '
                'for reliability and security.</li>'
                '<li><strong>Remote Communities:</strong> off-road community diesel '
                'generation cost.</li>'
                '</ul>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════════
    # TAB 2 — COST DRIVERS
    # ═══════════════════════════════════════════════════════════════
    with tab_drivers:
        _drv = enriched_df[enriched_df['Account'].apply(
            is_double_digit_excluding_multiples_of_10)].copy()
        _drv = _drv.sort_values('FOAK LCOE', ascending=False)
        _drv = _drv[_drv['FOAK LCOE'] >= 5]
        _drv = _drv.head(10)

        if _drv.empty:
            st.info('No accounts with FOAK LCOE >= 5 $/MWh found.')
        else:
            st.markdown(
                '<p style="color:#64748b;font-size:0.85rem;margin-bottom:1rem;">'
                'Per-account LCOE contributions ($/MWh) sorted by FOAK impact. '
                'Error bars show +/-1 standard deviation across Monte Carlo samples.</p>',
                unsafe_allow_html=True,
            )

            bar_width = 0.38
            r1 = np.arange(len(_drv))
            r2 = r1 + bar_width

            matplotlib.rcParams.update({
                'font.family':    'DejaVu Sans',
                'font.size':       13,
                'axes.titlesize':  15,
                'axes.labelsize':  13,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
            })

            fig, ax = plt.subplots(figsize=(max(13, len(_drv) * 1.6), 7))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#f8fafc')

            foak_err = _drv['FOAK LCOE_std'] if 'FOAK LCOE_std' in _drv.columns else None
            noak_err = _drv['NOAK LCOE_std'] if 'NOAK LCOE_std' in _drv.columns else None

            ax.bar(r1, _drv['FOAK LCOE'], width=bar_width,
                   color='#E05C2B', edgecolor='white', linewidth=0.8,
                   label='FOAK', zorder=3,
                   yerr=foak_err, capsize=5,
                   error_kw=dict(elinewidth=1.8, ecolor='#9a3412', capthick=1.8))
            ax.bar(r2, _drv['NOAK LCOE'], width=bar_width,
                   color='#1B7FBD', edgecolor='white', linewidth=0.8,
                   label='NOAK', zorder=3,
                   yerr=noak_err, capsize=5,
                   error_kw=dict(elinewidth=1.8, ecolor='#1155aa', capthick=1.8))

            ax.set_xticks(r1 + bar_width / 2)
            ax.set_xticklabels(_drv['Account Title'], rotation=35, ha='right',
                               fontsize=12, color='#1e293b', fontweight='500')
            ax.set_ylabel('LCOE Contribution ($/MWh)', fontsize=13,
                          color='#1e293b', labelpad=10)
            ax.yaxis.set_tick_params(labelcolor='#1e293b', labelsize=12)
            ax.set_xlim(-0.4, len(_drv) - 0.15)

            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.yaxis.grid(True, linestyle='--', linewidth=0.7,
                          alpha=0.7, color='#cbd5e1', zorder=0)
            ax.set_axisbelow(True)

            legend = ax.legend(fontsize=12, frameon=True, framealpha=1,
                               edgecolor='#e2e8f0', facecolor='white',
                               loc='upper right', handlelength=1.5,
                               borderpad=0.8, labelspacing=0.5)
            legend.get_frame().set_linewidth(1.0)

            plt.tight_layout(pad=2.0)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            st.image(buf, use_container_width=True)
            plt.close(fig)
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)

        # --- Detailed cost drivers (one level deeper) ---
        _det = detailed_sorted_df.copy() if not detailed_sorted_df.empty else pd.DataFrame()

        if _det.empty:
            st.info('No detailed accounts with FOAK LCOE >= 5 $/MWh found.')
        else:
            st.markdown('---')
            st.markdown(
                '<p style="color:#64748b;font-size:0.85rem;margin-bottom:1rem;">'
                '<strong>Detailed cost drivers</strong> — one level deeper than the chart above. '
                '3-digit sub-accounts are shown where available; otherwise the 2-digit parent is kept.</p>',
                unsafe_allow_html=True,
            )

            bar_width = 0.38
            r1 = np.arange(len(_det))
            r2 = r1 + bar_width

            matplotlib.rcParams.update({
                'font.family':    'DejaVu Sans',
                'font.size':       13,
                'axes.titlesize':  15,
                'axes.labelsize':  13,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
            })

            fig2, ax2 = plt.subplots(figsize=(max(13, len(_det) * 1.6), 7))
            fig2.patch.set_facecolor('white')
            ax2.set_facecolor('#f8fafc')

            foak_err2 = _det['FOAK LCOE_std'] if 'FOAK LCOE_std' in _det.columns else None
            noak_err2 = _det['NOAK LCOE_std'] if 'NOAK LCOE_std' in _det.columns else None

            ax2.bar(r1, _det['FOAK LCOE'], width=bar_width,
                    color='#E05C2B', edgecolor='white', linewidth=0.8,
                    label='FOAK', zorder=3,
                    yerr=foak_err2, capsize=5,
                    error_kw=dict(elinewidth=1.8, ecolor='#9a3412', capthick=1.8))
            ax2.bar(r2, _det['NOAK LCOE'], width=bar_width,
                    color='#1B7FBD', edgecolor='white', linewidth=0.8,
                    label='NOAK', zorder=3,
                    yerr=noak_err2, capsize=5,
                    error_kw=dict(elinewidth=1.8, ecolor='#1155aa', capthick=1.8))

            ax2.set_xticks(r1 + bar_width / 2)
            ax2.set_xticklabels(_det['Account Title'], rotation=35, ha='right',
                                fontsize=12, color='#1e293b', fontweight='500')
            ax2.set_ylabel('LCOE Contribution ($/MWh)', fontsize=13,
                           color='#1e293b', labelpad=10)
            ax2.yaxis.set_tick_params(labelcolor='#1e293b', labelsize=12)
            ax2.set_xlim(-0.4, len(_det) - 0.15)

            for spine in ['top', 'right', 'left']:
                ax2.spines[spine].set_visible(False)
            ax2.spines['bottom'].set_color('#cbd5e1')
            ax2.yaxis.grid(True, linestyle='--', linewidth=0.7,
                           alpha=0.7, color='#cbd5e1', zorder=0)
            ax2.set_axisbelow(True)

            legend2 = ax2.legend(fontsize=12, frameon=True, framealpha=1,
                                 edgecolor='#e2e8f0', facecolor='white',
                                 loc='upper right', handlelength=1.5,
                                 borderpad=0.8, labelspacing=0.5)
            legend2.get_frame().set_linewidth(1.0)

            plt.tight_layout(pad=2.0)
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', dpi=200, bbox_inches='tight', facecolor='white')
            buf2.seek(0)
            st.image(buf2, use_container_width=True)
            plt.close(fig2)
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    # ═══════════════════════════════════════════════════════════════
    # TAB 3 — FULL BREAKDOWN
    # ═══════════════════════════════════════════════════════════════
    with tab_table:
        st.markdown(
            f'<p style="color:#64748b;font-size:0.85rem;margin-bottom:1rem;">'
            f'Full cost breakdown by Code of Accounts. Highlighted rows are parent accounts. '
            f'All dollar values in {ESCALATION_YEAR} USD.</p>',
            unsafe_allow_html=True,
        )

        _foak_col = next((c for c in display_df.columns
                          if c.startswith('FOAK Estimated Cost (') and 'std' not in c), None)
        _noak_col = next((c for c in display_df.columns
                          if c.startswith('NOAK Estimated Cost (') and 'std' not in c), None)
        _foak_std = next((c for c in display_df.columns if 'FOAK Estimated Cost std' in c), None)
        _noak_std = next((c for c in display_df.columns if 'NOAK Estimated Cost std' in c), None)
        _have_lcoe = 'FOAK LCOE' in enriched_df.columns

        def _pm(mean_val, std_val):
            m = _fmt_table_val(mean_val)
            if m == '-':
                return '-'
            try:
                mn = float(mean_val)
                sd = float(std_val)
            except (TypeError, ValueError):
                return m
            if sd == 0 or np.isnan(sd):
                return m
            lo = _fmt_table_val(max(0, mn - sd))
            hi = _fmt_table_val(mn + sd)
            return f'{lo} – {hi}' if lo != hi else lo

        def _fmt_lcoe_tab(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return '-'
            if np.isnan(v) or v <= 0:
                return '-'
            if v < 1:
                return '< 1'
            if v < 10:
                return f'{v:.1f}'
            return str(int(round(v)))

        def _pm_lcoe(mean_val, std_val):
            m = _fmt_lcoe_tab(mean_val)
            if m == '-':
                return '-'
            try:
                mn = float(mean_val)
                sd = float(std_val)
            except (TypeError, ValueError):
                return m
            if np.isnan(sd) or sd <= 0:
                return m
            lo = _fmt_lcoe_tab(max(0.0, mn - sd))
            hi = _fmt_lcoe_tab(mn + sd)
            return lo if lo == hi else f'{lo} – {hi}'

        def _fmt_account(x):
            if isinstance(x, str):
                return x
            try:
                v = float(x)
                return str(int(v)) if v == int(v) else f'{v:g}'
            except (TypeError, ValueError):
                return str(x)

        table_df = display_df[['Account', 'Account Title']].copy()
        table_df['Account'] = table_df['Account'].apply(_fmt_account)

        _sf = display_df[_foak_std] if _foak_std else pd.Series('-', index=display_df.index)
        _sn = display_df[_noak_std] if _noak_std else pd.Series('-', index=display_df.index)
        table_df['FOAK Cost ($)'] = [_pm(m, s) for m, s in zip(display_df[_foak_col], _sf)]
        table_df['NOAK Cost ($)'] = [_pm(m, s) for m, s in zip(display_df[_noak_col], _sn)]

        if _have_lcoe:
            _ei   = display_df.index
            _e_fl  = enriched_df['FOAK LCOE'].reindex(_ei)
            _e_nl  = enriched_df['NOAK LCOE'].reindex(_ei)
            _e_fls = enriched_df['FOAK LCOE_std'].reindex(_ei) \
                     if 'FOAK LCOE_std' in enriched_df.columns else pd.Series(np.nan, index=_ei)
            _e_nls = enriched_df['NOAK LCOE_std'].reindex(_ei) \
                     if 'NOAK LCOE_std' in enriched_df.columns else pd.Series(np.nan, index=_ei)
            table_df['FOAK LCOE ($/MWh)'] = [_pm_lcoe(m, s) for m, s in zip(_e_fl, _e_fls)]
            table_df['NOAK LCOE ($/MWh)'] = [_pm_lcoe(m, s) for m, s in zip(_e_nl, _e_nls)]

        def _account_level(acct_str):
            try:
                v = float(str(acct_str).strip())
            except (ValueError, TypeError):
                return '-'
            ip = int(v)
            n  = len(str(ip))
            if n <= 2:
                return 0 if ip % 10 == 0 else 1
            elif n == 3:
                return 3 if v != ip else 2
            return 4

        _acct_levels = [_account_level(a) for a in table_df['Account']]
        _idx_to_pos  = {idx: pos for pos, idx in enumerate(table_df.index)}

        _EM = '\u2003'
        _PREFIX = {'-': '', 0: '', 1: f'{_EM}> ', 2: f'{_EM}{_EM}> ',
                   3: f'{_EM}{_EM}{_EM}. ', 4: f'{_EM}{_EM}{_EM}{_EM}. '}
        table_df['Account Title'] = [
            f"{_PREFIX.get(lv, _EM * 4)}{title}"
            for lv, title in zip(_acct_levels, display_df['Account Title'])
        ]

        _LEVEL_STYLE = {
            '-': ('background-color:#fef9c3', 'color:#78350f', 'font-weight:700'),
             0:  ('background-color:#1e3a5f', 'color:#f0f9ff', 'font-weight:700'),
             1:  ('background-color:#cfe2f3', 'color:#1a2e44', 'font-weight:600'),
             2:  ('background-color:#eaf4fb', 'color:#1a2e44', 'font-weight:500'),
             3:  ('background-color:#ffffff', 'color:#374151', 'font-weight:400'),
             4:  ('background-color:#ffffff', 'color:#374151', 'font-weight:400'),
        }

        def _row_style(row):
            lv = _acct_levels[_idx_to_pos[row.name]]
            bg, fg, fw = _LEVEL_STYLE.get(lv, ('background-color:#ffffff', 'color:#374151', ''))
            cell = f'{bg};{fg};{fw}'
            return [cell] * len(row)

        _num_cols = [c for c in table_df.columns if c not in ('Account', 'Account Title')]
        styled = (
            table_df.style
            .apply(_row_style, axis=1)
            .set_properties(subset=_num_cols, **{'text-align': 'right', 'padding': '2px 8px'})
            .set_table_styles([{
                'selector': 'thead tr th',
                'props': ('background-color:#1e3a5f;color:#f0f9ff;'
                          'font-weight:600;font-size:0.75rem;'
                          'text-transform:uppercase;letter-spacing:0.05em;')
            }])
        )

        _col_cfg = {
            'Account':       st.column_config.TextColumn(width='small'),
            'Account Title': st.column_config.TextColumn(width='medium'),
            'FOAK Cost ($)': st.column_config.TextColumn(width='small'),
            'NOAK Cost ($)': st.column_config.TextColumn(width='small'),
        }
        if _have_lcoe:
            _col_cfg['FOAK LCOE ($/MWh)'] = st.column_config.TextColumn(width='small')
            _col_cfg['NOAK LCOE ($/MWh)'] = st.column_config.TextColumn(width='small')

        st.dataframe(styled, use_container_width=True, height=580,
                     hide_index=True, column_config=_col_cfg)

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            display_df.to_excel(writer, index=False, sheet_name='Cost Estimate')
        buffer.seek(0)

        dl_col, _ = st.columns([1, 3])
        dl_col.download_button(
            label='⬇  Download Full Excel',
            data=buffer.getvalue(),
            file_name=f'MOUSE_cost_estimate_{reactor_type}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True,
        )

    st.markdown(
        """
        <div style='text-align: center; font-size: 0.9rem; color: gray; padding-top: 2rem; padding-bottom: 1rem;'>
            © 2025 Battelle Energy Alliance, LLC. MOUSE is released under the MIT License.
        </div>
        """,
        unsafe_allow_html=True,
    )