"""Preserve GCMR subcritical cases across a parametric-study file update.

The GCMR parametric-study file at
    assets/Ref_Results/GCMR_parametric_size_power_enrichment.xlsx
contains a 'Merged Data' sheet with one row per (Assembly Rings,
Core Rings, Active Height, Enrichment, Power MWt) configuration.
Subcritical cases (Fuel Lifetime = 0 or NaN) are intentional —
they encode where the design space goes subcritical.

When the file is updated upstream (e.g. to add a k_eff column),
the new file may not include these subcritical rows.  This script
preserves them across the update:

    1. Run BEFORE the update (extracts from current file):
         python tools/preserve_gcmr_subcritical.py extract
       Saves all subcritical rows to
         assets/Ref_Results/_gcmr_subcritical_cases.csv

       Or extract from a specific older Excel file:
         python tools/preserve_gcmr_subcritical.py extract /path/old.xlsx

    2. Replace the Excel file with the updated version.

    3. Run AFTER the update:
         python tools/preserve_gcmr_subcritical.py merge
       Re-injects the saved subcritical rows back into the updated
       Excel file, skipping any (NA, NC, H, E, P) configurations
       that are already present.

If the update has already happened and the old file is in git
history, you can recover the subcritical rows like this:

    git show <old_commit>:assets/Ref_Results/GCMR_parametric_size_power_enrichment.xlsx > /tmp/old_gcmr.xlsx
    python tools/preserve_gcmr_subcritical.py extract /tmp/old_gcmr.xlsx
    python tools/preserve_gcmr_subcritical.py merge

Run from the MOUSE repo root.
"""
import sys
from pathlib import Path

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent
_XLSX_PATH = _REPO_ROOT / 'assets' / 'Ref_Results' / 'GCMR_parametric_size_power_enrichment.xlsx'
_CSV_PATH  = _REPO_ROOT / 'assets' / 'Ref_Results' / '_gcmr_subcritical_cases.csv'
_SHEET     = 'Merged Data'

_KEY_COLS = ['Assembly Rings', 'Core Rings', 'Active Height',
             'Enrichment', 'Power MWt']


def extract(src_path=None):
    """Save subcritical rows (Fuel Lifetime = 0 or NaN) to a CSV.

    If src_path is None, reads the current file at _XLSX_PATH.
    Otherwise reads from the specified path (useful when extracting
    from an older Excel file recovered via git show).
    """
    src = Path(src_path) if src_path else _XLSX_PATH
    if not src.exists():
        sys.exit(f'ERROR: {src} not found.')

    print(f'Reading: {src}')
    df = pd.read_excel(src, sheet_name=_SHEET, engine='openpyxl')
    if 'Fuel Lifetime' not in df.columns:
        sys.exit("ERROR: 'Fuel Lifetime' column not found in the sheet.")

    mask = df['Fuel Lifetime'].isna() | (df['Fuel Lifetime'] == 0)
    sub = df.loc[mask].copy()

    print(f'Total rows in current file: {len(df)}')
    print(f'Subcritical rows (Fuel Lifetime = 0 or NaN): {len(sub)}')

    sub.to_csv(_CSV_PATH, index=False)
    print(f'\nSaved {len(sub)} subcritical rows to:')
    print(f'  {_CSV_PATH}')

    if not sub.empty:
        print('\nFirst few preserved rows:')
        print(sub[_KEY_COLS + ['Fuel Lifetime']].head(10).to_string(index=False))


def merge():
    """Re-inject saved subcritical rows into the (updated) Excel file."""
    if not _CSV_PATH.exists():
        sys.exit(f"ERROR: {_CSV_PATH} not found. Run 'extract' first.")
    if not _XLSX_PATH.exists():
        sys.exit(f'ERROR: {_XLSX_PATH} not found.')

    saved = pd.read_csv(_CSV_PATH)
    df = pd.read_excel(_XLSX_PATH, sheet_name=_SHEET, engine='openpyxl')

    # Find which (NA, NC, H, E, P) configurations the new file is
    # missing — those are the ones we need to re-inject.
    missing_key_cols = [c for c in _KEY_COLS if c not in df.columns]
    if missing_key_cols:
        sys.exit(f'ERROR: updated file is missing key columns: {missing_key_cols}')

    existing_keys = set(map(tuple, df[_KEY_COLS].values.tolist()))
    new_rows = saved[~saved.apply(
        lambda r: tuple(r[k] for k in _KEY_COLS) in existing_keys, axis=1
    )].copy()

    print(f'Updated file rows: {len(df)}')
    print(f'Saved subcritical rows: {len(saved)}')
    print(f'Subcritical rows to re-inject (not already present): {len(new_rows)}')

    if new_rows.empty:
        print('Nothing to merge — every saved subcritical row is already in the file.')
        return

    # Align columns: keep all columns from the updated file; add the
    # saved rows with their values, with NaN for any new columns
    # the updated file has but the CSV doesn't (e.g. k_eff).
    for col in df.columns:
        if col not in new_rows.columns:
            new_rows[col] = pd.NA
    new_rows = new_rows[df.columns]  # match column order

    merged = pd.concat([df, new_rows], ignore_index=True)

    # Write back, preserving any other sheets in the workbook.
    with pd.ExcelWriter(_XLSX_PATH, engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        merged.to_excel(writer, sheet_name=_SHEET, index=False)

    print(f'\nWrote {len(merged)} total rows to {_SHEET} in:')
    print(f'  {_XLSX_PATH}')
    print(f'  ({len(df)} from updated file + {len(new_rows)} re-injected)')


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('extract', 'merge'):
        sys.exit(
            'Usage:\n'
            '  python tools/preserve_gcmr_subcritical.py extract [optional_path.xlsx]\n'
            '  python tools/preserve_gcmr_subcritical.py merge'
        )

    if sys.argv[1] == 'extract':
        src = sys.argv[2] if len(sys.argv) >= 3 else None
        extract(src)
    else:
        merge()
