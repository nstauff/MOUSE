# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
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
    """Minimal stub for openmc.Material that stores density correctly."""

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
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from reactor_config import build_params, SubcriticalError
from cost.cost_estimation import bottom_up_cost_estimate, transform_dataframe
from cost.cost_drivers import cost_drivers_estimate, is_double_digit_excluding_multiples_of_10

# ---------------------------------------------------------------------------
# Performance patch: cache the per-row Excel read inside calculate_inflation_multiplier.
# Without this, pd.read_excel is called once per cost-database row on every run.
# ---------------------------------------------------------------------------
import functools
import cost.cost_escalation as _ce

_orig_inflation = _ce.calculate_inflation_multiplier

@functools.lru_cache(maxsize=512)
def _cached_inflation_multiplier(file_path, base_dollar_year, cost_type, escalation_year):
    return _orig_inflation(file_path, base_dollar_year, cost_type, escalation_year)

_ce.calculate_inflation_multiplier = _cached_inflation_multiplier

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
        'main': (
            os.path.join(_ASSETS, 'GCMR_Core.png'),
            'GCMR core cross-section — hexagonal fuel assemblies containing TRISO fuel '
            'compacts arranged in a honeycomb pattern, cooled by helium gas flowing '
            'through dedicated coolant channels, with graphite reflector and control drums.',
        ),
        'details': [
            (
                os.path.join(_ASSETS, 'GCMR_Core (zoomed in).png'),
                'GCMR core zoomed — detailed view of the hexagonal assembly arrangement '
                'showing the compact-to-assembly packing and inter-assembly helium flow paths.',
            ),
            (
                os.path.join(_ASSETS, 'GCMR_Fuel Assembly.png'),
                'GCMR fuel assembly cross-section — TRISO fuel compacts, helium coolant '
                'channels, and ZrH moderator booster pins embedded in a graphite matrix.',
            ),
            (
                os.path.join(_ASSETS, 'GCMR_Fuel Assembly (zoomed in).png'),
                'GCMR fuel assembly zoomed — individual TRISO particles visible within '
                'the graphite fuel compact at the target packing fraction.',
            ),
            (
                os.path.join(_ASSETS, 'GCMR_TRISO_Particle.png'),
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
# Returns: (display_df, enriched_df_raw, params)
#   display_df     — transformed (integer costs) table for display and Excel export,
#                    includes FOAK LCOE / NOAK LCOE columns from cost_drivers_estimate
#   enriched_df_raw — raw float version of the enriched table, used for the plot
#   params         — fully-populated params dict
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _run_estimate(reactor_type, power_mwt, enrichment, interest_rate,
                  construction_duration, debt_to_equity, operation_mode,
                  emergency_shutdowns, startup_duration,
                  tax_credit_type, tax_credit_value):
    overrides = {
        'Interest Rate': interest_rate,
        'Construction Duration': construction_duration,
        'Debt To Equity Ratio': debt_to_equity,
        'Escalation Year': 2024,
        'Operation Mode': operation_mode,
        'Emergency Shutdowns Per Year': emergency_shutdowns,
        'Startup Duration after Emergency Shutdown': startup_duration,
    }
    if tax_credit_type == 'PTC':
        overrides['PTC credit value'] = tax_credit_value
        overrides['PTC credit period'] = 10
        overrides['Tax Rate'] = 0.21
        # Bonus multipliers are already baked into the selected PTC credit value.
    elif tax_credit_type == 'ITC':
        overrides['ITC credit level'] = tax_credit_value

    p = build_params(reactor_type, power_mwt, enrichment, overrides)
    raw_df = bottom_up_cost_estimate('cost/Cost_Database.xlsx', p)

    # Enrich with per-account LCOE contributions (no PNG — plotting key not set).
    enriched_df = cost_drivers_estimate(raw_df, p)

    return transform_dataframe(enriched_df), enriched_df, p


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title='MOUSE — Microreactor Cost Estimator',
    page_icon='⚛',
    layout='wide',
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title('MOUSE — Microreactor Cost Estimator')
st.caption(
    'Microreactor Online Unified Simulation Engine · '
    'Bottom-up capital & levelized cost estimates for microreactor designs · '
    'All costs in **2024 USD**.'
)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header('Inputs')

    reactor_label = st.selectbox(
        'Reactor Type',
        options=list(_REACTOR_LABELS.values()),
        help='Select a microreactor design to estimate costs for.',
    )
    reactor_type = _LABEL_TO_KEY[reactor_label]

    st.markdown('**Core Design Parameters**')

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
    power_mwt = st.slider(
        'Thermal Power (MWt)',
        min_value=1,
        max_value=20,
        value=_power_defaults[reactor_type],
        step=1,
        key=f'power_{reactor_type}',
        help='Thermal power output. Affects power-dependent params and fuel lifetime via interpolation.',
    )

    st.divider()
    st.markdown('**Economic Parameters**')

    interest_rate = st.number_input(
        'Interest Rate (%)',
        min_value=0.0,
        max_value=30.0,
        value=7.0,
        step=0.5,
        format='%.1f',
        help='Annual interest rate used in financing cost calculations.',
    )

    construction_duration = st.number_input(
        'Construction Duration (months)',
        min_value=1,
        max_value=120,
        value=12,
        step=1,
        help='Number of months from ground-break to commercial operation.',
    )

    debt_to_equity = st.slider(
        'Debt-to-Equity Ratio',
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        format='%.2f',
        help='Fraction of capital funded by debt (0 = all equity, 1 = all debt).',
    )

    operation_mode = st.selectbox(
        'Operation Mode',
        options=['Autonomous', 'Non-Autonomous'],
        help='Autonomous = minimal on-site operators; Non-Autonomous = full staffing.',
    )

    emergency_shutdowns = st.number_input(
        'Emergency Shutdowns per Year',
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1,
        format='%.1f',
        help='Expected number of unplanned emergency shutdowns per year.',
    )

    startup_duration = st.number_input(
        'Startup Duration after Emergency Shutdown (days)',
        min_value=1,
        max_value=365,
        value=14,
        step=1,
        help='Days required to restart the reactor after an emergency shutdown.',
    )

    st.divider()
    st.markdown('**Government Subsidy (IRA Tax Credits)**')

    tax_credit_type = st.selectbox(
        'Tax Credit',
        options=['None', 'PTC', 'ITC'],
        index=0,
        help=(
            'None: no tax credit applied. '
            'PTC: Production Tax Credit (reduces LCOE). '
            'ITC: Investment Tax Credit (reduces capital cost).'
        ),
    )

    tax_credit_value = None
    if tax_credit_type == 'PTC':
        tax_credit_value = st.selectbox(
            'PTC Credit Value ($/MWh)',
            options=[3.0, 3.3, 3.6, 15.0, 16.5, 18.0],
            index=3,
            format_func=lambda x: f'${x:.1f}/MWh',
            help=(
                'Total PTC value including any applicable IRA bonus multipliers '
                '(domestic content, energy community). '
                'Base rate: $3/MWh; full prevailing-wage rate: $15/MWh.'
            ),
        )
    elif tax_credit_type == 'ITC':
        tax_credit_value = st.selectbox(
            'ITC Credit Level',
            options=[0.06, 0.30, 0.40, 0.50],
            index=1,
            format_func=lambda x: f'{x*100:.0f}%',
            help='ITC as a fraction of overnight capital cost (OCC).',
        )

    run_button = st.button('Run Cost Estimate', type='primary', use_container_width=True)

# ---------------------------------------------------------------------------
# Reactor design section — always visible
# ---------------------------------------------------------------------------
main_img, main_caption = _REACTOR_IMAGES[reactor_type]['main']
st.subheader(f'{reactor_label} — Core Design')
st.image(main_img, use_container_width=True)
st.caption(main_caption)

with st.expander('View more design details'):
    for img_path, img_caption in _REACTOR_IMAGES[reactor_type]['details']:
        st.image(img_path, use_container_width=True)
        st.caption(img_caption)
        st.divider()

# ---------------------------------------------------------------------------
# Placeholder for results
# ---------------------------------------------------------------------------
if not run_button:
    st.info('Configure inputs in the sidebar and click **Run Cost Estimate** to begin.')
    st.stop()

# ---------------------------------------------------------------------------
# Build params and run cost estimate
# ---------------------------------------------------------------------------
with st.spinner('Running cost estimate…'):
    try:
        display_df, enriched_df, params = _run_estimate(
            reactor_type, power_mwt, enrichment,
            interest_rate / 100.0, construction_duration, debt_to_equity,
            operation_mode, emergency_shutdowns, startup_duration,
            tax_credit_type, tax_credit_value,
        )
    except SubcriticalError as exc:
        st.warning('### Reactor is Subcritical')
        st.error(str(exc))
        col_a, col_b, col_c = st.columns(3)
        col_a.metric('Fuel Lifetime', '0 days')
        col_b.metric('Thermal Power', f'{power_mwt} MWt')
        col_c.metric('Enrichment', f'{enrichment * 100:.2f}%')
        st.info(
            'No cost estimate is available for a subcritical operating point. '
            'Try reducing the power or increasing the enrichment.'
        )
        st.stop()
    except Exception as exc:
        st.error(f'Cost estimation failed: {exc}')
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# ---------------------------------------------------------------------------
# Helper: extract a single cost value from the display dataframe
# ---------------------------------------------------------------------------
def _get_value(df, account, which='FOAK'):
    """Return the cost for a given account label (FOAK or NOAK), or NaN."""
    row = df[df['Account'] == account]
    if row.empty:
        return float('nan')
    prefix = 'FOAK Estimated Cost (' if which == 'FOAK' else 'NOAK Estimated Cost ('
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return float('nan')
    val = row[cols[0]].iloc[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return float('nan')


# ---------------------------------------------------------------------------
# Summary metrics — FOAK and NOAK side by side
# ---------------------------------------------------------------------------
st.subheader('Summary')

fuel_lifetime = params.get('Fuel Lifetime', float('nan'))
power_mwe     = params.get('Power MWe', float('nan'))

occ_foak  = _get_value(display_df, 'OCC',  'FOAK')
tci_foak  = _get_value(display_df, 'TCI',  'FOAK')
lcoe_foak = _get_value(display_df, 'LCOE', 'FOAK')
occ_noak  = _get_value(display_df, 'OCC',  'NOAK')
tci_noak  = _get_value(display_df, 'TCI',  'NOAK')
lcoe_noak = _get_value(display_df, 'LCOE', 'NOAK')

# Row 1: FOAK values + reactor info
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('FOAK OCC',  f'${occ_foak:,.0f}'  if not pd.isna(occ_foak)  else 'N/A')
c2.metric('FOAK TCI',  f'${tci_foak:,.0f}'  if not pd.isna(tci_foak)  else 'N/A')
c3.metric('FOAK LCOE', f'${lcoe_foak:,.1f} /MWh' if not pd.isna(lcoe_foak) else 'N/A')
c4.metric('Power Output', f'{power_mwe:.2f} MWe')
c5.metric('Fuel Lifetime', f'{fuel_lifetime:,} days' if not pd.isna(float(fuel_lifetime)) else 'N/A')

# Row 2: NOAK values
n1, n2, n3 = st.columns(3)
n1.metric('NOAK OCC',  f'${occ_noak:,.0f}'  if not pd.isna(occ_noak)  else 'N/A')
n2.metric('NOAK TCI',  f'${tci_noak:,.0f}'  if not pd.isna(tci_noak)  else 'N/A')
n3.metric('NOAK LCOE', f'${lcoe_noak:,.1f} /MWh' if not pd.isna(lcoe_noak) else 'N/A')

# Tax-credit adjusted metrics (if applicable)
if tax_credit_type == 'ITC':
    occ_itc  = _get_value(display_df, 'OCC (ITC-adjusted)',  'FOAK')
    tci_itc  = _get_value(display_df, 'TCI (ITC-adjusted)',  'FOAK')
    lcoe_itc = _get_value(display_df, 'LCOE (ITC-adjusted)', 'FOAK')
    st.caption('With ITC applied:')
    ci1, ci2, ci3 = st.columns(3)
    ci1.metric('OCC (ITC-adjusted)',  f'${occ_itc:,.0f}'       if not pd.isna(occ_itc)  else 'N/A')
    ci2.metric('TCI (ITC-adjusted)',  f'${tci_itc:,.0f}'       if not pd.isna(tci_itc)  else 'N/A')
    ci3.metric('LCOE (ITC-adjusted)', f'${lcoe_itc:,.1f} /MWh' if not pd.isna(lcoe_itc) else 'N/A')
elif tax_credit_type == 'PTC':
    lcoe_ptc = _get_value(display_df, 'LCOE with PTC', 'FOAK')
    st.caption('With PTC applied:')
    st.metric('LCOE with PTC', f'${lcoe_ptc:,.1f} /MWh' if not pd.isna(lcoe_ptc) else 'N/A')

# ---------------------------------------------------------------------------
# Cost drivers plot
# ---------------------------------------------------------------------------
st.subheader('Cost Drivers')

_drv = enriched_df[enriched_df['Account'].apply(is_double_digit_excluding_multiples_of_10)].copy()
_drv = _drv.sort_values('FOAK LCOE', ascending=False)
_drv = _drv[_drv['FOAK LCOE'] >= 5]

if _drv.empty:
    st.info('No accounts with FOAK LCOE ≥ 5 $/MWh found.')
else:
    bar_width = 0.35
    r1 = np.arange(len(_drv))
    r2 = r1 + bar_width

    fig, ax = plt.subplots(figsize=(max(10, len(_drv) * 1.4), 6))
    ax.bar(r1, _drv['FOAK LCOE'], width=bar_width, color='orangered', edgecolor='black',
           label=f'FOAK {reactor_type}',
           yerr=_drv['FOAK LCOE_std'] if 'FOAK LCOE_std' in _drv.columns else None,
           capsize=6, error_kw=dict(elinewidth=1.5))
    ax.bar(r2, _drv['NOAK LCOE'], width=bar_width, color='royalblue', edgecolor='black',
           label=f'NOAK {reactor_type}',
           yerr=_drv['NOAK LCOE_std'] if 'NOAK LCOE_std' in _drv.columns else None,
           capsize=6, error_kw=dict(elinewidth=1.5))
    ax.set_xticks(r1 + bar_width / 2)
    ax.set_xticklabels(_drv['Account Title'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('LCOE Contribution ($/MWh)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='dashed', linewidth=0.5)
    ax.minorticks_on()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Full cost table
# ---------------------------------------------------------------------------
st.subheader('Detailed Cost Breakdown')

def _highlight_parents(row):
    acct = row.get('Account', None)
    try:
        if float(acct) == int(float(acct)) and float(acct) % 10 == 0:
            return ['background-color: #f0f2f6; font-weight: bold'] * len(row)
    except (TypeError, ValueError):
        pass
    return [''] * len(row)


styled = display_df.style.apply(_highlight_parents, axis=1)
st.dataframe(styled, use_container_width=True, height=600)

# ---------------------------------------------------------------------------
# Download button
# ---------------------------------------------------------------------------
st.subheader('Export Results')

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    display_df.to_excel(writer, index=False, sheet_name='Cost Estimate')
buffer.seek(0)

st.download_button(
    label='Download Excel',
    data=buffer.getvalue(),
    file_name=f'MOUSE_cost_estimate_{reactor_type}.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
)
