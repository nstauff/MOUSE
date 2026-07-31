# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""
Enrichment sensitivity study for GCMR Design A.
Sweeps enrichment from 15% to 20% and plots LCOE (FOAK and NOAK).
"""

import contextlib
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from .def_watts_exec_GCMR_Design_A import gcmr_calc


@contextlib.contextmanager
def redirect_all_output(log_file):
    """Redirect Python-level and C-level stdout/stderr to a log file.
    contextlib.redirect_stdout only hooks sys.stdout; OpenMC writes at the
    OS file-descriptor level, so we must also redirect fd 1 and fd 2 with
    os.dup2 to suppress those messages from the terminal.
    """
    with open(log_file, 'w') as f:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = f
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

# **************************************************************************************************************************
#                                           Enrichment Sensitivity Study (15% to 20%)
# **************************************************************************************************************************
enrichment_values = [0.1975] #np.linspace(0.1975, 0.90, 4)
moderator_options1 = {
    'Moderator Booster':   'YHx',
    'Moderator Liner':   'Nb',
    'Moderator Envelope':   'SiC',
}
moderator_options2 = {
    'Moderator Booster':   'Graphite',
    'Moderator Liner':   'Graphite',
    'Moderator Envelope':   'Graphite',
}
moderator_options3 = {
    'Moderator Booster':   'YHx',
    'Moderator Liner':   'FeCrAl',
    'Moderator Envelope':   'FeCrAl',
}
moderator_scan = [
    ('YHx-Nb-SiC',   moderator_options1),
    ('Graphite',     moderator_options2),
    ('YHx-FeCrAl',   moderator_options3),
]

base_params = {
    'Moderator Liner Thickness': 0.01,
    'Moderator Envelope Thickness': 0.04,
    'Moderator Envelope Radius': 0.60,
    'Packing Fraction':    0.4,
    'Compact Fuel Radius': 1.0,
    'Active Height': 250,
}

lcoe_foak = {label: [] for label, _ in moderator_scan}
lcoe_noak = {label: [] for label, _ in moderator_scan}

for mod_label, mod_params in moderator_scan:
    for enr in enrichment_values:
        varied_params = {**base_params, **mod_params, 'Enrichment': enr}
        calc_id = f"sensitivity_{mod_label}_enr{enr:.4f}"
        log_file = f"NS_TESTS/log_{calc_id}.txt"
        print(f"Running {calc_id} (log -> {log_file})")
        with redirect_all_output(log_file):
            results = gcmr_calc(varied_params, calc_id)
        temp_coeff, sd_margin, fuel_lifetime, max_peak_factor, lcoe_foak_val, lcoe_noak_val = results
        lcoe_foak[mod_label].append(lcoe_foak_val)
        lcoe_noak[mod_label].append(lcoe_noak_val)
        print(
            f"Moderator: {mod_label} | "
            f"Enrichment: {enr:.2%} | "
            f"Temp Coeff: {temp_coeff:.2f} pcm/K | "
            f"SD Margin: {sd_margin:.0f} pcm | "
            f"Fuel Lifetime: {fuel_lifetime:.1f} days | "
            f"Max Peak Factor: {max_peak_factor:.3f} | "
            f"LCOE FOAK: {lcoe_foak_val:.2f} $/MWh | "
            f"LCOE NOAK: {lcoe_noak_val:.2f} $/MWh"
        )

# Plot
enrichment_pct = np.array(enrichment_values) * 100
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.tab10(np.arange(len(moderator_scan)))
for color, (mod_label, _) in zip(colors, moderator_scan):
    ax.plot(enrichment_pct, lcoe_foak[mod_label], 'o-',  color=color, label=f'{mod_label} FOAK')
    ax.plot(enrichment_pct, lcoe_noak[mod_label], 's--', color=color, label=f'{mod_label} NOAK')
ax.set_xlabel('Enrichment (%)')
ax.set_ylabel('LCOE ($/MWh)')
ax.set_title('LCOE vs. Enrichment — GCMR Design A')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('NS_TESTS/LCOE_vs_enrichment_GCMR.png', dpi=150)
plt.show()
print("Plot saved to NS_TESTS/LCOE_vs_enrichment_GCMR.png")
