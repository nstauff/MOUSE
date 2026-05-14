# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
One-time extraction of material densities (g/cm3) from
core_design.openmc_materials_database into a JSON lookup table.

Why: the webapp imports the full materials-database build chain just to
read densities for mass calculations. Loading the full mix_materials
pipeline costs base RAM and accumulates per-Run leak from pandas/matplotlib
internals that ride along. Replacing the runtime build with a JSON lookup
eliminates that import chain entirely.

Run this script ONCE from the MOUSE repo root:

    /usr/bin/python3 webapp/extract_materials_densities.py

(No openmc needed  the script stubs it inline, same way webapp/app.py
does. The stub stores set_density values correctly per project notes.)

It writes webapp/materials_densities.json. Commit that file. A later
refactor swaps the runtime call to collect_materials_data(...) for a
JSON lookup keyed by reactor type + material name.

Materials whose density is set in units other than g/cm3 (atom/b-cm for
homog_TRISO, heatpipe, etc.) are skipped: per project notes, they never
contribute to mass calculations.
"""

import json
import os
import sys
from unittest.mock import MagicMock

# Make sure MOUSE root is on sys.path so core_design imports resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Stub openmc BEFORE importing core_design (same approach as webapp/app.py).
# This stub adds .density_units tracking so we can filter atom/b-cm materials
# out of the JSON output.
# ---------------------------------------------------------------------------
class _MaterialStub:
    def __init__(self, name=None, temperature=None):
        self.name = name
        self.temperature = temperature
        self.density = 0.0
        self.density_units = None

    def set_density(self, units, value):
        self.density = value
        self.density_units = units

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
            unit_set = {getattr(m, 'density_units', None) for m in materials}
            if unit_set == {'g/cm3'}:
                result.density_units = 'g/cm3'
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

# Now safe to import core_design.
from core_design.openmc_materials_database import collect_materials_data  # noqa: E402


# Reference params per reactor. Values copied from the watts_exec_*.py
# example files. Hardcoded constant densities in the database don't depend
# on these; only mix_materials() outputs do.
REFERENCE_PARAMS = {
    'LTMR': {
        'Enrichment': 0.1975,
        'H_Zr_ratio': 1.6,
        'er_wo': 0,
        'U_met_wo': 0.3,
        'Common Temperature': 600,
    },
    'GCMR': {
        'Enrichment': 0.1975,
        'UO2 atom fraction': 0.7,
        'Common Temperature': 850,
    },
    'HPMR': {
        'Enrichment': 0.19985,
        'Common Temperature': 1000,
    },
}


def main():
    output = {}
    for reactor, params in REFERENCE_PARAMS.items():
        print(f"\n--- {reactor} (params: {params}) ---")
        materials_db = collect_materials_data(params)
        reactor_entries = {}
        skipped = []
        for name, mat in materials_db.items():
            units = getattr(mat, 'density_units', None)
            density = getattr(mat, 'density', None)
            if units == 'g/cm3' and density:
                reactor_entries[name] = float(density)
            else:
                skipped.append(f"{name} (units={units})")
        output[reactor] = reactor_entries
        print(f"  extracted {len(reactor_entries)} g/cm3 densities")
        if skipped:
            print(f"  skipped: {', '.join(skipped)}")

    out_path = os.path.join(os.path.dirname(__file__), 'materials_densities.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, sort_keys=True)

    print(f"\nWrote {out_path}")
    total = sum(len(v) for v in output.values())
    print(f"Total entries across reactors: {total}")


if __name__ == '__main__':
    main()
