# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""Precompute per-state industrial retail price distributions from EIA-861.

Input : cost_drivers_and_markets/Sales_Ult_Cust_<year>.xlsx
Output: assets/Ref_Results/state_industrial_price_distribution_<year>.csv

Per state we report:
  n_utilities  - utility-state rows contributing
  weighted_mean - sum(revenue) / sum(sales), the EIA Table 5.6.A definition
  min, p10, q1, median, q3, p90, max - empirical distribution of
    per-utility prices (revenue / sales for each row)

All prices in cents/kWh. Filters applied:
  - drop Utility Number 99999 (EIA state-level imputation row)
  - keep Service Type == 'Bundled' (full retail price, generation + T&D)
  - drop rows with non-positive Industrial Revenue or Sales

Regenerate by running this script with the openmc-env conda env active:
  python cost_drivers_and_markets/build_state_industrial_price_distribution.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

YEAR = 2024
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / 'cost_drivers_and_markets' / f'Sales_Ult_Cust_{YEAR}.xlsx'
DST = REPO_ROOT / 'assets' / 'Ref_Results' / (
    f'state_industrial_price_distribution_{YEAR}.csv'
)

UNUM = 'Utility Characteristics__Utility Number'
STYPE = 'Utility Characteristics__Service Type'
STATE = 'Utility Characteristics__State'
REV_I = 'INDUSTRIAL__Revenues__Thousand Dollars'
SAL_I = 'INDUSTRIAL__Sales__Megawatthours'


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        '__'.join([str(s) for s in col if 'Unnamed' not in str(s)]).strip()
        for col in df.columns
    ]
    return df


def _summarize(group: pd.DataFrame) -> pd.Series:
    prices = group['cents_per_kWh']
    return pd.Series({
        'n_utilities':   int(len(group)),
        'weighted_mean': (group[REV_I].sum() * 100) / group[SAL_I].sum(),
        'min':           prices.min(),
        'p10':           prices.quantile(0.10),
        'q1':            prices.quantile(0.25),
        'median':        prices.median(),
        'q3':            prices.quantile(0.75),
        'p90':           prices.quantile(0.90),
        'max':           prices.max(),
    })


def main() -> None:
    raw = pd.read_excel(SRC, sheet_name='States', header=[0, 1, 2])
    raw = _flatten_columns(raw)

    raw[REV_I] = pd.to_numeric(raw[REV_I], errors='coerce')
    raw[SAL_I] = pd.to_numeric(raw[SAL_I], errors='coerce')

    mask = (
        (raw[UNUM] != 99999)
        & (raw[STYPE] == 'Bundled')
        & (raw[REV_I] > 0)
        & (raw[SAL_I] > 0)
    )
    df = raw.loc[mask].copy()
    df['cents_per_kWh'] = df[REV_I] * 100 / df[SAL_I]

    summary = (df.groupby(STATE, group_keys=False)
                 .apply(_summarize, include_groups=False)
                 .reset_index()
                 .rename(columns={STATE: 'state'})
                 .sort_values('state'))
    summary['n_utilities'] = summary['n_utilities'].astype(int)

    DST.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DST, index=False, float_format='%.4f')

    n_states = len(summary)
    total_rows = len(df)
    print(f'Read    : {SRC.relative_to(REPO_ROOT)}')
    print(f'Wrote   : {DST.relative_to(REPO_ROOT)}')
    print(f'Source rows after filters : {total_rows}')
    print(f'States covered            : {n_states}')
    print()
    print(summary.to_string(index=False,
                            float_format=lambda v: f'{v:7.2f}'
                            if isinstance(v, float) else str(v)))


if __name__ == '__main__':
    main()
