# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
Generate reference cross-section PNGs for the GCMR webapp.

Produces:
  assets/Ref_openmc_2d_designs/GCMR_core_NA{N_A}_NC{N_C}.png
      — one per (N_A, N_C) pair the webapp slider exposes:
        (4,3), (4,4), (4,5), (5,3), (5,4), (5,5),
        (6,3), (6,4), (6,5), (7,3), (7,4), (7,5)            12 files

  assets/Ref_openmc_2d_designs/GCMR_fuel_assembly_NA{N_A}.png
      — one per N_A in {4, 5, 6, 7}                          4 files

  assets/Ref_openmc_2d_designs/GCMR_TRISO_particle.png       1 file
  assets/Ref_openmc_2d_designs/GCMR_fuel_assembly_zoomed.png 1 file

Total: 18 PNGs.

Run from the MOUSE repo root in an env that has OpenMC installed:
    cd /Users/botros/projects/MOUSE
    conda activate openmc-env
    python assets/generate_gcmr_reference_plots.py

Note: this script does NOT run a Monte Carlo simulation — only the
matplotlib-based geometry plotter is used. No cross-section data is
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

from core_design.openmc_template_GCMR import build_openmc_model_GCMR
from core_design.drums import (
    calculate_drums_volumes_and_masses,
    calculate_reflector_mass_GCMR,
    calculate_moderator_mass_GCMR,
)


_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'assets', 'Ref_openmc_2d_designs')
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# Match the webapp's GCMR slider: 4 N_A × 3 N_C = 12 pairs.
NA_NC_PAIRS = [(na, nc) for na in (4, 5, 6, 7) for nc in (3, 4, 5)]


def _build_minimal_gcmr_params(n_a, n_c, active_height_cm=215.0):
    """Minimum params dict required to build GCMR OpenMC geometry + plots.

    The values for power, enrichment etc. don't affect the geometry image —
    the cross-section is the same regardless of the operating point.
    """
    params = {
        # --- materials ---
        'reactor type': 'GCMR',
        'TRISO Fueled': 'Yes',
        'Fuel': 'UCO',
        'Enrichment': 0.1975,
        'UO2 atom fraction': 0.7,
        'Radial Reflector': 'Graphite',
        'Axial Reflector': 'Graphite',
        'Matrix Material': 'Graphite',
        'Moderator': 'Graphite',
        'Moderator Booster Materials': ['ZrH'],
        'Coolant': 'Helium',
        'Common Temperature': 850,
        'Control Drum Absorber': 'B4C_enriched',
        'Control Drum Reflector': 'Graphite',
        'HX Material': 'SS316',
        'Power MWt': 1.0,
        # --- geometry ---
        'Fuel Pin Materials': ['UCO', 'buffer_graphite', 'PyC', 'SiC', 'PyC'],
        'Fuel Pin Radii': [0.0250, 0.0350, 0.0390, 0.0425, 0.0465],
        'Compact Fuel Radius': 0.6225,
        'Packing Fraction': 0.3,
        'Coolant Channel Radius': 0.35,
        'Moderator Booster Radii': [0.55],
        'Lattice Pitch': 2.25,
        'Assembly Rings': int(n_a),
        'Core Rings': int(n_c),
        'Active Height': float(active_height_cm),
        # --- drums (radius and reflector thickness auto-resolved) ---
        'Drum Absorber Thickness': 1,
        # --- run-time toggles ---
        'plotting': 'Y',
        'cross_sections_xml_location': 'cross_sections.xml',
    }
    params['Assembly FTF'] = params['Lattice Pitch'] * (params['Assembly Rings'] - 1) * np.sqrt(3)
    # Drums.calculate_drums_volumes_and_masses auto-resolves Drum Radius,
    # Radial Reflector Thickness, Axial Reflector Thickness, Core Radius,
    # and Drum Height for GCMR.
    calculate_drums_volumes_and_masses(params)
    calculate_reflector_mass_GCMR(params)
    calculate_moderator_mass_GCMR(params)
    return params


def _move(src, dst):
    """Move a file, raising FileNotFoundError if the source is missing."""
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Expected plot output {src!r} was not produced. "
            f"The build_openmc_model_GCMR function may have changed its "
            f"output filenames — check core_design/openmc_template_GCMR.py."
        )
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)


def main():
    # Generate one core plot per (N_A, N_C) pair, plus one fuel-assembly
    # plot per N_A (assembly layout depends on N_A only). The TRISO
    # particle and zoomed fuel assembly plots don't depend on geometry —
    # generated once.
    triso_saved        = False
    assembly_zoom_saved = False
    assembly_per_na_saved = set()   # which N_A values already have an assembly png

    for (n_a, n_c) in NA_NC_PAIRS:
        print(f"\n=== Building GCMR core for (N_A={n_a}, N_C={n_c}) ===")
        params = _build_minimal_gcmr_params(n_a, n_c)
        cwd_before = os.getcwd()
        work_dir = os.path.join(_REPO_ROOT, f"_tmp_gcmr_plot_NA{n_a}_NC{n_c}")
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        try:
            build_openmc_model_GCMR(params)
        finally:
            os.chdir(cwd_before)

        # Per-(N_A, N_C) core image
        _move(
            os.path.join(work_dir, 'Core.png'),
            os.path.join(_OUTPUT_DIR, f'GCMR_core_NA{n_a}_NC{n_c}.png'),
        )

        # Per-N_A fuel-assembly image (save once per N_A)
        if n_a not in assembly_per_na_saved:
            _move(
                os.path.join(work_dir, 'Fuel Assembly.png'),
                os.path.join(_OUTPUT_DIR, f'GCMR_fuel_assembly_NA{n_a}.png'),
            )
            assembly_per_na_saved.add(n_a)

        # TRISO particle and zoomed fuel-assembly: same for all geometries —
        # save once.
        if not triso_saved:
            _move(
                os.path.join(work_dir, 'TRISO_Particle.png'),
                os.path.join(_OUTPUT_DIR, 'GCMR_TRISO_particle.png'),
            )
            triso_saved = True
        if not assembly_zoom_saved:
            _move(
                os.path.join(work_dir, 'Fuel Assembly (zoomed in).png'),
                os.path.join(_OUTPUT_DIR, 'GCMR_fuel_assembly_zoomed.png'),
            )
            assembly_zoom_saved = True

        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nAll plots written to: {_OUTPUT_DIR}")


if __name__ == '__main__':
    main()
