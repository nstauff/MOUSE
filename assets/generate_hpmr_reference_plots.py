# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
Generate reference cross-section PNGs for the HPMR webapp.

The HPMR parametric study has N_A (Number of Rings per Assembly)
locked at 6 — only N_C (Number of Rings per Core) varies across
{3, 4, 5, 6, 7}.  So the webapp's slider exposes 5 distinct core
geometries; each gets its own PNG.

Produces:
  assets/Ref_openmc_2d_designs/HPMR_core_NA6_NC{N_C}.png
      — one per N_C in {3, 4, 5, 6, 7}                        5 files

  assets/Ref_openmc_2d_designs/HPMR_fuel_assembly_NA6.png
      — single file (assembly layout depends on N_A only,
        and N_A is locked at 6 in the parametric study)        1 file

  assets/Ref_openmc_2d_designs/HPMR_fuel_pin_universe.png
      — fuel-pin lattice cell, geometry-independent            1 file

  assets/Ref_openmc_2d_designs/HPMR_heatpipe_universe.png
      — heat-pipe lattice cell, geometry-independent           1 file

Total: 8 PNGs.

Run from the MOUSE repo root in an env that has OpenMC installed:
    cd /Users/botros/projects/MOUSE
    conda activate openmc-env
    python assets/generate_hpmr_reference_plots.py

Note: this script does NOT run a Monte Carlo simulation — only the
matplotlib-based geometry plotter is used.  No cross-section data is
loaded; the cross_sections.xml path is just written into materials.xml
as a string and never read.
"""
import os
import shutil
import sys

# Run from repo root so 'core_design.*' / 'webapp.*' imports resolve
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from core_design.openmc_template_HPMR import build_openmc_model_HPMR
from core_design.utils import (
    calculate_number_fuel_elements_hpmr,
    number_of_heatpipes_hmpr,
)
from core_design.drums import (
    calculate_drums_volumes_and_masses,
    calculate_reflector_and_moderator_mass_HPMR,
)


_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'assets', 'Ref_openmc_2d_designs')
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# HPMR slider: N_A locked at 6, N_C ∈ {3, 4, 5, 6, 7}.  5 unique cores.
NA_FIXED = 6
NC_VALUES = [3, 4, 5, 6, 7]


def _build_minimal_hpmr_params(n_a, n_c, active_height_cm=None):
    """Minimum params dict required to build HPMR OpenMC geometry + plots.

    The values for power, enrichment, height etc. don't affect the
    geometry image — the cross-section is the same regardless of the
    operating point or active height.
    """
    if active_height_cm is None:
        # Use the parametric study's middle-band height for this N_C
        # (purely cosmetic — only matters for the side-view, not the
        # cross-section plots this script generates).
        active_height_cm = {3: 271, 4: 379, 5: 488, 6: 596, 7: 704}.get(int(n_c), 300)

    params = {
        # --- materials ---
        'reactor type': 'HPMR',
        'TRISO Fueled': 'Yes',
        'Fuel': 'homog_TRISO',
        'Enrichment': 0.1975,
        'Radial Reflector': 'Graphite',
        'Axial Reflector': 'Graphite',
        'Moderator': 'monolith_graphite',
        'Coolant': 'Helium',
        'Control Drum Absorber': 'B4C_natural',
        'Control Drum Reflector': 'Graphite',
        'Cooling Device': 'heatpipe',
        'Common Temperature': 1000,
        'HX Material': 'SS316',
        'Power MWt': 5.0,
        # --- geometry ---
        'Fuel Pin Materials':  ['homog_TRISO', 'Helium'],
        'Fuel Pin Radii':      [1.00, 1.05],
        'Heat Pipe Materials': ['heatpipe', 'Helium'],
        'Heat Pipe Radii':     [1.10, 1.15],
        'Number of Rings per Assembly': int(n_a),
        'Number of Rings per Core':     int(n_c),
        'Lattice Pitch': 3.4,
        'Active Height': float(active_height_cm),
        # --- drums (Drum Radius, Reflector, Core Radius, Drum Height
        # all auto-resolved by calculate_drums_volumes_and_masses) ---
        'Drum Count': 12,
        'Drum Absorber Thickness': 1,
        # --- run-time toggles ---
        'plotting': 'Y',
        'cross_sections_xml_location': 'cross_sections.xml',
    }

    # Same geometry-derivation chain as _build_hpmr in reactor_config.py
    params['Assembly FTF'] = ((params['Lattice Pitch'] * (params['Number of Rings per Assembly'] - 1)
                               + 1.4 * params['Fuel Pin Radii'][-1]) * np.sqrt(3))
    params['hexagonal Core Edge Length'] = ((params['Assembly FTF'] * (params['Number of Rings per Core'] - 1))
                                            + (params['Assembly FTF'] / 2) + 6.6)
    params['Fuel Pin Count per Assembly'] = calculate_number_fuel_elements_hpmr(
        params['Number of Rings per Assembly']
    )
    params['Fuel Assemblies Count'] = ((3 * params['Number of Rings per Core'] ** 2)
                                       - (3 * params['Number of Rings per Core']))
    params['Fuel Pin Count'] = (params['Fuel Assemblies Count']
                                * params['Fuel Pin Count per Assembly'])
    number_of_heatpipes_hmpr(params)

    # Auto-resolve Drum Radius / Reflector / Core Radius / Drum Height
    calculate_drums_volumes_and_masses(params)
    calculate_reflector_and_moderator_mass_HPMR(params)
    return params


def _move(src, dst):
    """Move a file, raising FileNotFoundError if the source is missing."""
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Expected plot output {src!r} was not produced. "
            f"The build_openmc_model_HPMR function may have changed its "
            f"output filenames — check core_design/openmc_template_HPMR.py."
        )
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)


def main():
    # Generate one core plot per N_C.  Fuel pin, heat pipe, and fuel
    # assembly cross-sections are saved once (they depend on N_A only,
    # which is locked at 6).
    fuel_pin_saved   = False
    heatpipe_saved   = False
    assembly_saved   = False

    for n_c in NC_VALUES:
        n_a = NA_FIXED
        print(f"\n=== Building HPMR core for (N_A={n_a}, N_C={n_c}) ===")
        params = _build_minimal_hpmr_params(n_a, n_c)

        cwd_before = os.getcwd()
        work_dir = os.path.join(_REPO_ROOT, f"_tmp_hpmr_plot_NC{n_c}")
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        try:
            build_openmc_model_HPMR(params)
        finally:
            os.chdir(cwd_before)

        # Per-(N_A, N_C) core image — name kept consistent with the
        # GCMR convention so the webapp's image-switch code works the
        # same way (HPMR_core_NA{na}_NC{nc}.png).
        _move(
            os.path.join(work_dir, 'core.png'),
            os.path.join(_OUTPUT_DIR, f'HPMR_core_NA{n_a}_NC{n_c}.png'),
        )

        # Geometry-independent plots — save once on the first iteration.
        if not assembly_saved:
            _move(
                os.path.join(work_dir, 'fuel_assembly.png'),
                os.path.join(_OUTPUT_DIR, f'HPMR_fuel_assembly_NA{n_a}.png'),
            )
            assembly_saved = True
        if not fuel_pin_saved:
            _move(
                os.path.join(work_dir, 'fuel_pin_universe.png'),
                os.path.join(_OUTPUT_DIR, 'HPMR_fuel_pin_universe.png'),
            )
            fuel_pin_saved = True
        if not heatpipe_saved:
            _move(
                os.path.join(work_dir, 'heatpipe_universe.png'),
                os.path.join(_OUTPUT_DIR, 'HPMR_heatpipe_universe.png'),
            )
            heatpipe_saved = True

        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nAll plots written to: {_OUTPUT_DIR}")


if __name__ == '__main__':
    main()
