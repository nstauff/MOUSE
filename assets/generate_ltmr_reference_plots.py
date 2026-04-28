# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
Generate reference cross-section PNGs for the LTMR webapp.

Produces:
  assets/Ref_openmc_2d_designs/LTMR_core_N{N}.png    for N = 10..24 (15 files)
  assets/Ref_openmc_2d_designs/LTMR_fuel_pin_universe.png        (1 file)
  assets/Ref_openmc_2d_designs/LTMR_moderator_pin_universe.png   (1 file)

The pin and moderator universes don't depend on N — generated once.

Run from the MOUSE repo root in an env that has OpenMC installed:
    cd /Users/botros/projects/MOUSE
    conda activate openmc-env       # (or whatever your OpenMC env is named)
    python assets/generate_ltmr_reference_plots.py

Each plot takes a few seconds; the full run is a few minutes.

Note: this script does NOT run a Monte Carlo simulation — only the
matplotlib-based geometry plotter is used. No cross-section data is
loaded; the cross_sections.xml path is just written into materials.xml
as a string and never read.
"""
import os
import shutil
import sys
import warnings

# Run from repo root so 'core_design.*' / 'webapp.*' imports resolve
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from core_design.openmc_template_LTMR import build_openmc_model_LTMR
from core_design.utils import (
    calculate_lattice_radius,
    calculate_hex_apothem,
    calculate_pins_in_assembly,
)
from core_design.drums import (
    calculate_drums_volumes_and_masses,
    calculate_reflector_mass_LTMR,
    calculate_moderator_mass,
)
from core_design.pins_arrangement import LTMR_pins_arrangement


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'assets', 'Ref_openmc_2d_designs')
os.makedirs(_OUTPUT_DIR, exist_ok=True)


def _build_minimal_ltmr_params(n_rings, active_height_cm=80.0):
    """Minimum params dict required to build LTMR OpenMC geometry + plots.

    The values for power, enrichment etc. don't affect the geometry image
    — the cross-section is the same regardless of the operating point.
    """
    params = {
        # --- materials (same as webapp/_build_ltmr) ---
        'reactor type': 'LTMR',
        'TRISO Fueled': 'No',
        'Fuel': 'TRIGA_fuel',
        'H_Zr_ratio': 1.6,
        'U_met_wo': 0.3,
        'er_wo': 0,
        'Coolant': 'NaK',
        'Radial Reflector': 'Graphite',
        'Axial Reflector': 'Graphite',
        'Moderator': 'ZrH',
        'Control Drum Absorber': 'B4C_enriched',
        'Control Drum Reflector': 'Graphite',
        'Common Temperature': 600,
        'HX Material': 'SS316',
        'Enrichment': 0.1975,           # placeholder
        'Power MWt': 1.0,               # placeholder
        # --- geometry ---
        'Fuel Pin Materials': ['Zr', None, 'TRIGA_fuel', None, 'SS304'],
        'Fuel Pin Radii':     [0.28575, 0.3175, 1.5113, 1.5367, 1.5875],
        'Moderator Pin Materials': ['ZrH', 'SS304'],
        'Moderator Pin Inner Radius': 1.5367,
        'Moderator Pin Radii': [1.5367, 1.5875],
        'Pin Gap Distance': 0.1,
        'Pins Arrangement': LTMR_pins_arrangement,
        'Number of Rings per Assembly': int(n_rings),
        'Active Height': float(active_height_cm),
        # --- drums (count fixed at 18 to match parametric study) ---
        'Number of Drums': 18,
        'Drum Absorber Thickness': 1,
        'Drum Absorber Arc Degrees': 120,
        # --- run-time toggles ---
        'plotting': 'Y',
        # Cross-sections.xml path is written into materials.xml but never
        # actually read — universe.plot() is matplotlib-based and does
        # not transport particles. Any non-empty string works.
        'cross_sections_xml_location': 'cross_sections.xml',
    }
    params['Lattice Radius']       = calculate_hex_apothem(params)
    params['Assembly FTF']         = 2 * params['Lattice Radius']
    params['Fuel Pin Count']       = calculate_pins_in_assembly(params, 'FUEL')
    params['Moderator Pin Count']  = calculate_pins_in_assembly(params, 'MODERATOR')
    params['Moderator Mass']       = calculate_moderator_mass(params)
    return params


def _move(src, dst):
    """Move a file, overwriting any existing destination."""
    if not os.path.exists(src):
        warnings.warn(f"Expected output {src!r} not produced — skipping rename.")
        return
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)


def main():
    n_values = list(range(10, 25))   # 10, 11, …, 24

    # Generate one core plot per N. The fuel-pin-universe and
    # moderator-pin-universe plots don't depend on N, so we keep only the
    # ones from the first (smallest-N) run.
    pin_universe_saved = False

    for n in n_values:
        print(f"\n=== Building LTMR core for N = {n} ===")
        params = _build_minimal_ltmr_params(n)
        cwd_before = os.getcwd()
        # build_openmc_model_LTMR writes plot PNGs into the current
        # working directory.  Use a per-N temp dir so files don't collide.
        work_dir = os.path.join(_REPO_ROOT, f"_tmp_ltmr_plot_N{n}")
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        try:
            build_openmc_model_LTMR(params)
        finally:
            os.chdir(cwd_before)

        # Move the core plot to the per-N reference name
        _move(
            os.path.join(work_dir, 'core.png'),
            os.path.join(_OUTPUT_DIR, f'LTMR_core_N{n}.png'),
        )

        if not pin_universe_saved:
            _move(
                os.path.join(work_dir, 'fuel_pin_universe.png'),
                os.path.join(_OUTPUT_DIR, 'LTMR_fuel_pin_universe.png'),
            )
            _move(
                os.path.join(work_dir, 'moderator_pin_universe.png'),
                os.path.join(_OUTPUT_DIR, 'LTMR_moderator_pin_universe.png'),
            )
            pin_universe_saved = True

        # Clean up the temp dir (and any OpenMC XMLs left behind)
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nAll plots written to: {_OUTPUT_DIR}")


if __name__ == '__main__':
    main()
