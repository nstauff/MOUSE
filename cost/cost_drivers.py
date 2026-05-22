# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cost.code_of_account_processing import get_estimated_cost_column


def is_double_digit_excluding_multiples_of_10(val):
    """Returns True for double-digit account numbers that are not multiples of 10
    (e.g. 21, 22, 75 — but not 10, 20, 30). These are the sub-accounts used as
    cost drivers in the bar chart."""
    return (
        isinstance(val, (int, float))
        and val == int(val)
        and 10 <= int(val) <= 99
        and int(val) % 10 != 0
    )


def is_three_digit_excluding_multiples_of_10(val):
    """Returns True for three-digit integer account numbers that are not multiples of 10
    (e.g. 111, 221, 753 — but not 110, 220, 750). These are one level below the
    double-digit accounts used in the standard cost drivers chart."""
    return (
        isinstance(val, (int, float))
        and val == int(val)
        and 100 <= int(val) <= 999
        and int(val) % 10 != 0
    )


def get_detailed_driver_rows(df, two_digit_accounts):
    """
    Build the detailed cost drivers dataset — one level below the standard 2-digit accounts.

    For each 2-digit account:
    - If it has 3-digit children in df, include those children.
    - If it has no 3-digit children (leaf node), keep the 2-digit account itself.

    Parameters
    ----------
    df : pd.DataFrame
        The enriched cost table (output of cost_drivers_estimate), containing LCOE columns.
    two_digit_accounts : list
        Account numbers at the 2-digit level to expand (e.g. [11, 12, 21, 22, ...]).

    Returns
    -------
    pd.DataFrame
        Rows representing one level deeper than the standard cost drivers.
    """
    three_digit_df = df[df['Account'].apply(is_three_digit_excluding_multiples_of_10)]

    result_indices = []
    for acct in two_digit_accounts:
        children = three_digit_df[
            three_digit_df['Account'].apply(lambda x, a=acct: int(x) // 10 == int(a))
        ]
        if not children.empty:
            result_indices.extend(children.index.tolist())
        else:
            # No 3-digit children — keep the 2-digit account row itself
            result_indices.extend(df[df['Account'] == acct].index.tolist())

    return df.loc[result_indices]


def energy_cost_levelized_per_acct(params, capital_cost, ann_cost):
    """
    Compute the LCOE contribution ($/MWh) of a single account.

    Pass the account's cost as `capital_cost` for accounts 10–69 (one-time capital),
    or as `ann_cost` for accounts 70–89 (recurring annual costs). Pass 0 for the
    other argument. Both arguments may be scalars or pandas Series.

    This mirrors the same discounted-cash-flow logic used by energy_cost_levelized
    in non_direct_cost.py but operates on a single account at a time so the
    contribution of each cost driver to the total LCOE can be isolated.
    """
    sum_cost = 0
    sum_elec = 0
    for i in range(1 + params['Levelization Period']):
        if i == 0:
            year_cap  = capital_cost
            year_ann  = 0
            year_elec = 0
        else:
            year_cap  = 0
            year_ann  = ann_cost
            year_elec = params['Power MWe'] * params['Capacity Factor'] * 365 * 24  # MWh
        discount = (1 + params['Discount Rate']) ** i
        sum_cost += (year_cap + year_ann) / discount
        sum_elec += year_elec / discount
    return sum_cost / sum_elec


def cost_drivers_estimate(df, params):
    """
    Compute per-account LCOE contributions and produce a FOAK vs NOAK bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        The detailed cost table returned by bottom_up_cost_estimate (before
        integer conversion).  Must contain the FOAK and NOAK cost columns as
        well as the corresponding std columns when Number of Samples > 1.
    params : dict
        The standard MOUSE params dictionary.

    Output
    ------
    Saves  <reactor_type>_cost_drivers.png  in the working directory.
    """
    foak_col     = get_estimated_cost_column(df, 'F')
    noak_col     = get_estimated_cost_column(df, 'N')
    foak_std_col = foak_col.replace('Cost', 'Cost std')
    noak_std_col = noak_col.replace('Cost', 'Cost std')

    have_std = foak_std_col in df.columns and noak_std_col in df.columns

    # Capital accounts (1xx–6xx): cost paid at year 0 → capital_cost argument
    # Annual  accounts (7xx–8xx): cost paid each year → ann_cost argument
    mask_annual  = df['Account'].astype(str).str[0].isin(['7', '8'])
    mask_capital = df['Account'].astype(str).str[0].isin(['1', '2', '3', '4', '5', '6'])

    df = df.copy()

    # --- Mean LCOE per account ---
    df['FOAK LCOE'] = np.nan
    df['NOAK LCOE'] = np.nan
    df.loc[mask_capital, 'FOAK LCOE'] = energy_cost_levelized_per_acct(
        params, df.loc[mask_capital, foak_col], 0)
    df.loc[mask_capital, 'NOAK LCOE'] = energy_cost_levelized_per_acct(
        params, df.loc[mask_capital, noak_col], 0)
    df.loc[mask_annual,  'FOAK LCOE'] = energy_cost_levelized_per_acct(
        params, 0, df.loc[mask_annual, foak_col])
    df.loc[mask_annual,  'NOAK LCOE'] = energy_cost_levelized_per_acct(
        params, 0, df.loc[mask_annual, noak_col])

    # --- Std of LCOE per account (propagated from cost std) ---
    df['FOAK LCOE_std'] = np.nan
    df['NOAK LCOE_std'] = np.nan
    if have_std:
        df.loc[mask_capital, 'FOAK LCOE_std'] = energy_cost_levelized_per_acct(
            params, df.loc[mask_capital, foak_std_col], 0)
        df.loc[mask_capital, 'NOAK LCOE_std'] = energy_cost_levelized_per_acct(
            params, df.loc[mask_capital, noak_std_col], 0)
        df.loc[mask_annual,  'FOAK LCOE_std'] = energy_cost_levelized_per_acct(
            params, 0, df.loc[mask_annual, foak_std_col])
        df.loc[mask_annual,  'NOAK LCOE_std'] = energy_cost_levelized_per_acct(
            params, 0, df.loc[mask_annual, noak_std_col])

    # --- Filter to the meaningful sub-accounts and sort ---
    filtered_df = df[df['Account'].apply(is_double_digit_excluding_multiples_of_10)]
    sorted_df   = filtered_df.sort_values(by='FOAK LCOE', ascending=False)
    sorted_df   = sorted_df[sorted_df['FOAK LCOE'] >= 5]   # drop negligible contributors

    # --- Plot (only when requested) ---
    reactor_type = params.get('reactor type', 'Reactor')

    if params.get('plotting') == "Y":
        if sorted_df.empty:
            print("Warning: No accounts with FOAK LCOE >= 5 $/MWh found. Skipping cost drivers plot.")
        else:
            foak_errors = sorted_df['FOAK LCOE_std'] if have_std else None
            noak_errors = sorted_df['NOAK LCOE_std'] if have_std else None

            bar_width       = 0.35
            r1              = np.arange(len(sorted_df))
            r2              = r1 + bar_width
            error_bar_props = dict(capsize=15, elinewidth=3)

            plt.figure(figsize=(22, 20))
            plt.bar(r1, sorted_df['FOAK LCOE'], color='orangered', width=bar_width,
                    edgecolor='black', label=f'FOAK {reactor_type}',
                    yerr=foak_errors, error_kw=error_bar_props)
            plt.bar(r2, sorted_df['NOAK LCOE'], color='royalblue', width=bar_width,
                    edgecolor='black', label=f'NOAK {reactor_type}',
                    yerr=noak_errors, error_kw=error_bar_props)

            plt.xticks(r1 + bar_width / 2, sorted_df['Account Title'],
                       rotation=45, ha='right', fontsize=24)
            plt.xlim(-0.2, len(sorted_df) - 0.5)
            plt.grid(axis='y', which='both', color='grey', linestyle='dashed', linewidth=0.5)
            plt.minorticks_on()
            plt.legend(loc='upper right', fontsize=38, frameon=True, edgecolor='black')
            plt.ylabel('LCOE ($/MWh)', fontsize=40)
            plt.yticks(fontsize=32)
            plt.tight_layout()

            output_path = f'{reactor_type}_cost_drivers.png'
            plt.savefig(output_path)
            plt.close()
            print(f"Cost drivers plot saved at {output_path}")

    # --- Detailed cost drivers (one level deeper) ---
    two_digit_accounts = filtered_df['Account'].tolist()
    detailed_df = get_detailed_driver_rows(df, two_digit_accounts)
    detailed_sorted_df = detailed_df.sort_values(by='FOAK LCOE', ascending=False)
    detailed_sorted_df = detailed_sorted_df[detailed_sorted_df['FOAK LCOE'] >= 5]

    if params.get('plotting') == "Y":
        if detailed_sorted_df.empty:
            print("Warning: No detailed accounts with FOAK LCOE >= 5 $/MWh. Skipping detailed cost drivers plot.")
        else:
            foak_errors = detailed_sorted_df['FOAK LCOE_std'] if have_std else None
            noak_errors = detailed_sorted_df['NOAK LCOE_std'] if have_std else None

            bar_width       = 0.35
            r1              = np.arange(len(detailed_sorted_df))
            r2              = r1 + bar_width
            error_bar_props = dict(capsize=15, elinewidth=3)

            plt.figure(figsize=(22, 20))
            plt.bar(r1, detailed_sorted_df['FOAK LCOE'], color='orangered', width=bar_width,
                    edgecolor='black', label=f'FOAK {reactor_type}',
                    yerr=foak_errors, error_kw=error_bar_props)
            plt.bar(r2, detailed_sorted_df['NOAK LCOE'], color='royalblue', width=bar_width,
                    edgecolor='black', label=f'NOAK {reactor_type}',
                    yerr=noak_errors, error_kw=error_bar_props)

            plt.xticks(r1 + bar_width / 2, detailed_sorted_df['Account Title'],
                       rotation=45, ha='right', fontsize=24)
            plt.xlim(-0.2, len(detailed_sorted_df) - 0.5)
            plt.grid(axis='y', which='both', color='grey', linestyle='dashed', linewidth=0.5)
            plt.minorticks_on()
            plt.legend(loc='upper right', fontsize=38, frameon=True, edgecolor='black')
            plt.ylabel('LCOE ($/MWh)', fontsize=40)
            plt.yticks(fontsize=32)
            plt.tight_layout()

            output_path = f'{reactor_type}_cost_drivers(detailed).png'
            plt.savefig(output_path)
            plt.close()
            print(f"Detailed cost drivers plot saved at {output_path}")

    return df, detailed_sorted_df