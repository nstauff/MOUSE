# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""
This script performs a bottom-up cost estimate for a Gas Cooled Microreactor (GCMR).
Parallel screening case: E5_enr_15p0.
OpenMC is used for core design calculations, and other Balance of Plant components are estimated.

Callable wrapper around examples/watts_exec_GCMR_A_E5_enr_15p_5y.py, for use by
NS_TESTS/sensitivity.py. The E5 design is kept intact — only the enrichment and the
moderator booster layers are driven from the caller. All the explicit geometry the
E5 case pins down (Active Height, Assembly FTF, Core Radius, reflector thicknesses,
drum and shutdown rod dimensions) is left untouched, since the geometry helpers now
validate that set for mutual consistency.
"""

import numpy as np
import watts  # Simulation workflows for one or multiple codes
from core_design.openmc_template_GCMR import *
from core_design.utils import *
from core_design.drums import *
from reactor_engineering_evaluation.fuel_calcs import fuel_calculations
from reactor_engineering_evaluation.BOP import *
from reactor_engineering_evaluation.vessels_calcs import *
from reactor_engineering_evaluation.tools import *
from cost.cost_estimation import detailed_bottom_up_cost_estimate

import warnings
warnings.filterwarnings("ignore")

import time

params = watts.Parameters()

def update_params(updates):
    params.update(updates)

# **************************************************************************************************************************
#                                                Sec. 0: Settings
# **************************************************************************************************************************
update_params({
    'plotting': "Y",  # "Y" or "N": Yes or No
    # MAC
#    'cross_sections_xml_location': '/Users/nstauff/Documents/TOOLS/OpenMC/lib80x_hdf5/cross_sections.xml',
#    'simplified_chain_thermal_xml': '/Users/nstauff/Documents/TOOLS/OpenMC/chain_endfb80_pwr.xml'
    # Linux - CP1
    'cross_sections_xml_location': '/home/nstauff/PROCEDURES/OpenMC/lib/e81/endfb-viii.1b2-hdf5/cross_sections.xml',
    'simplified_chain_thermal_xml': '/home/nstauff/PROCEDURES/OpenMC/lib/chain_endfb80_pwr.xml'
    # INL HPC
#    'cross_sections_xml_location': '/projects/MRP_MOUSE/openmc_data/endfb-viii.0-hdf5/cross_sections.xml',
#    'simplified_chain_thermal_xml': '/projects/MRP_MOUSE/openmc_data/simplified_thermal_chain11.xml'


})

def gcmr_calc(list_varied_params, calc_id, output_dir="NS_TESTS"):
    """execute GCMR E5 cases"""
    time_start = time.time()
    print("Varied Parameters:")
    for key, value in list_varied_params.items():
        print(f"  {key}: {value}")
    # **************************************************************************************************************************
    #                                                Sec. 1: Materials
    # **************************************************************************************************************************
    update_params({
        'reactor type': "GCMR",  # LTMR or GCMR
        'TRISO Fueled': "Yes",
        'Fuel': 'UCO',
        'Enrichment': list_varied_params['Enrichment'],  # The enrichment is a fraction. It has to be between 0 and 1
        'UO2 atom fraction': 0.7,  # Mixing UO2 and UC by atom fraction
        'Radial Reflector': 'Graphite',
        'Axial Reflector': 'Graphite',
        'Matrix Material': 'Graphite',  # matrix material is the background material within the compact fuel element between TRISO particles
        'Moderator': 'Graphite',  # the moderator is outside the compact fuel region
        # Three concentric booster layers (booster / liner / envelope), innermost -> outermost.
        # The E5 case itself uses a single ['ZrH'] layer at r = 0.5 cm.
        'Moderator Booster Materials': [list_varied_params['Moderator Booster'], list_varied_params['Moderator Liner'], list_varied_params['Moderator Envelope']],
        'Coolant': 'Helium',
        'Common Temperature': 850,  # Kelvins
        # IG-110 proxy: mean of axial/transverse CTE values in
        # ORNL/TM-2017/705, Table 2.2 (4.5 and 4.2 microstrain/K).
        'Graphite Linear Expansion Coefficient': 4.3e-6,  # 1/K
        'Control Drum Absorber': 'B4C_enriched',  # The absorber material in the control drums
        'Control Drum Reflector': 'Graphite',  # The reflector material in the control drums
        'Shutdown Rod Absorber': 'B4C_enriched',
        'Shutdown Rod Cladding': 'SS304',
        'HX Material': 'SS316',
    })

    # **************************************************************************************************************************
    #                                           Sec. 2: Geometry: Fuel Pins, Moderator Pins, Coolant, Hexagonal Lattice
    # **************************************************************************************************************************

    # Booster layer radii, built inwards from the outermost envelope radius so the
    # booster pin footprint stays at the value the E5 geometry was tuned around.
    envelope_radius = list_varied_params['Moderator Envelope Radius']
    liner_thickness = list_varied_params['Moderator Liner Thickness']
    envelope_thickness = list_varied_params['Moderator Envelope Thickness']

    update_params({
        # fuel pin details
        'Fuel Pin Materials': ['UCO', 'buffer_graphite', 'PyC', 'SiC', 'PyC'],
        'Fuel Pin Radii': [0.0250, 0.0350, 0.0390, 0.0425, 0.0465],  # cm # https://art.inl.gov/NRC%20Training%202019/04_TRISO_Fuel.pdf
        'Compact Fuel Radius': 0.6225,  # cm # The radius of the area that is occupied by the TRISO particles (fuel compact/ fuel element)
        'Packing Fraction': 0.4,
        'TRISO Packing Seed': 1,

        # Coolant channel and booster dimensions
        'Coolant Channel Radius': 0.35,  # cm
        'Moderator Booster Radii': [envelope_radius - envelope_thickness - liner_thickness, envelope_radius - envelope_thickness, envelope_radius],  # cm
        'Lattice Pitch': 2.25,
        'Assembly Rings': 6,
        'Core Rings': 5,

        # Central assembly
        'Central Shutdown Rod Radius': 0.85,  # cm
        'Central Shutdown Rod Clad Radius': 1.05,  # cm; 0.20 cm SS304
        'Central Shutdown Rod Ring': 2,
        'Central Shutdown Rod Count': 12,

        # Six assemblies surrounding the center
        'Surrounding Shutdown Rod Radius': 0.45,  # cm
        'Surrounding Shutdown Rod Clad Radius': 0.65,  # cm; 0.20 cm SS304
        'Surrounding Shutdown Rod Ring': 2,
        'Surrounding Shutdown Rod Count': 2,
        'Surrounding Shutdown Assembly Count': 6,

        # Explicit geometry values for this design. The geometry helper validates
        # these values and does not replace them with calculated dimensions.
        'Assembly FTF': 19.48557158514987,  # cm
        'Active Height': 200.0,  # cm
        'Radial Reflector Thickness': 9.742785792574935,  # cm
        'Axial Reflector Thickness': 9.742785792574935,  # cm
        'Core Radius': 107.17064371832429,  # cm
        'Shutdown Rod Height': 200.0,  # cm
    })

    # **************************************************************************************************************************
    #                                           Sec. 3: Control Drums
    # **************************************************************************************************************************
    update_params({
        'Drum Count': 24,
        'Drum Radius': 9.530986101432001,  # cm
        'Drum Tube Radius': 9.742785792574935,  # cm
        'Drum Absorber Thickness': 1, # cm
        'Drum Absorber Arc Degrees': 120.0,
        'Drum Height': 219.48557158514987,  # cm
        })
    calculate_drums_volumes_and_masses(params)
    calculate_gcmr_shutdown_rods_volumes_and_masses(params)
    calculate_reflector_mass_GCMR(params)
    calculate_moderator_mass_GCMR(params)

    # **************************************************************************************************************************
    #                                           Sec. 4: Overall System
    # **************************************************************************************************************************
    update_params({
        'Power MWt': 15,  # MWt
        'Thermal Efficiency': 0.4,
        'Heat Flux Criteria': 0.9,  # MW/m^2 (needs review)
        'Burnup Steps': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0,
                         30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0]  # MWd_per_Kg
        })

    params['Power MWe'] = params['Power MWt'] * params['Thermal Efficiency']
    params['Heat Flux'] = calculate_heat_flux_TRISO(params) # MW/m^2

    # **************************************************************************************************************************
    #                                           Sec. 5: Running OpenMC
    # **************************************************************************************************************************

    # --- Shutdown Margin (SDM) ---
    # When True, an additional OpenMC simulation is run with all control drums rotated
    # to the fully inserted (ARI - All Rods In) position. The SDM is then calculated
    # as the difference in reactivity (in pcm) between the ARO and ARI configurations.
    # A positive SDM means the reactor can be safely shut down with all drums inserted.
    # Recommended: True for final design verification; can be set to False to save
    # computation time during early design exploration.
    params['Shutdown Margin Calc'] = True  # True or False
    params['Cold Shutdown Temperature'] = 300  # K

    # --- Isothermal Temperature Coefficient ---
    # When True, two additional OpenMC simulations are run: one at 'Common Temperature'
    # and one at 'Common Temperature' + 'Temperature Perturbation'. The temperature
    # coefficient is then calculated in units of pcm/K.
    # A negative coefficient indicates the reactor is self-stabilizing (desired behavior).
    # Recommended: True for safety analysis; can be set to False to save computation time.
    params['Isothermal Temperature Coefficients'] = True  # True or False

    # --- Temperature Perturbation ---
    # The temperature step (in Kelvin) used for the isothermal temperature coefficient calculation.
    # Must be large enough to produce a keff difference above OpenMC Monte Carlo statistical
    # noise, but small enough to stay in the linear reactivity regime.
    # Typical range: 50–300 K. 100 K is chosen here as a balance between accuracy and
    # avoiding nonlinear effects.
    # Units: Kelvin
    # This parameter is REQUIRED only when 'Isothermal Temperature Coefficients' is True.
    params['Temperature Perturbation'] = 100  # K

    heat_flux_monitor = monitor_heat_flux(params)
    run_openmc(build_openmc_model_GCMR, heat_flux_monitor, params)
    fuel_calculations(params)  # calculate the fuel mass and SWU

    # --- Previously calculated OpenMC results ---
    # To bypass OpenMC later, comment out run_openmc(...) above and uncomment these assignments.
    # params['Fuel Lifetime'] = 1786  # days
    # params['Mass U235'] = 85655.7587486539  # g
    # params['Mass U238'] = 484226.2801659319  # g
    # params['Uranium Mass'] = 569.8820389145857  # kg

    # **************************************************************************************************************************
    #                                         Sec. 6: Primary Loop + Balance of Plant
    # **************************************************************************************************************************
    params.update({
        'Primary Loop Purification': True,
        'Secondary HX Mass': 0,
        'Compressor Pressure Ratio': 4,
        'Compressor Isentropic Efficiency': 0.8,
        'Primary Loop Count': 2,  # number of primary coolant loops in the plant
        'Primary Loop per loop load fraction': 0.5,  # each loop handles an equal share of the total load
        'Primary Loop Inlet Temperature': 300 + 273.15, # K
        'Primary Loop Outlet Temperature': 550 + 273.15, # K
        'Secondary Loop Inlet Temperature': 270 + 273.15, # K — cold-end PCHE pinch 30°C with 300°C primary inlet (was 290 -> 10°C pinch, below realistic PCHE design)
        'Secondary Loop Outlet Temperature': 500 + 273.15, # K,
        'Primary Loop Pressure Drop': 50e3,  # Pa — estimated assumption
    })
    params['Primary HX Mass'] = calculate_heat_exchanger_mass(params)  # Kg
    # calculate coolant mass flow rate
    mass_flow_rate(params)
    compressor_power(params)

    # Update BoP Parameters
    params.update({
        'BoP Count': 2,  # number of BoP systems in the plant
        'BoP per loop load fraction': 0.5,  # each BoP handles an equal share of the total load
        })
    params['BoP Power kWe'] = 1000 * params['Power MWe'] * params['BoP per loop load fraction']

    # Integrated Heat Transfer Vessel
    params.update({
        'Integrated Heat Transfer Vessel Thickness': 0, # cm
        'Integrated Heat Transfer Vessel Material': 'SA508',
    })
    GCMR_integrated_heat_transfer_vessel(params)

    # **************************************************************************************************************************
    #                                           Sec. 7 : Shielding
    # **************************************************************************************************************************
    update_params({
        'In Vessel Shield Thickness': 0,  # cm (no shield in vessel for GCMR)
        'In Vessel Shield Inner Radius': params['Core Radius'],
        'In Vessel Shield Material': 'B4C_natural',
        'Out Of Vessel Shield Thickness': 39.37,  # cm
        'Out Of Vessel Shield Material': 'WEP',
        'Out Of Vessel Shield Effective Density Factor': 0.5
    })
    params['In Vessel Shield Outer Radius'] = params['Core Radius'] + params['In Vessel Shield Thickness']

    # **************************************************************************************************************************
    #                                           Sec. 8 : Vessels Calculations
    # **************************************************************************************************************************
    update_params({
        'Vessel Radius': params['Core Radius'] + params['In Vessel Shield Thickness'],
        'Vessel Thickness': 3,  # cm — ASME Sec III Div 1 thin-shell with 4 MPa He, R=60-100 cm, S=138 MPa SA-508 at 350°C, +3 mm corrosion (was 1, below ASME pressure-driven minimum)
        'Vessel Lower Plenum Height': 30,  # cm — GA MHTGR / HTR-PM-class flow distributor (was 2.848, unit-conv bug)
        'Vessel Upper Plenum Height': 47.152,       # cm — outlet plenum for hot-leg gas exit
        'Vessel Upper Gas Gap': 0,
        'Vessel Bottom Depth': 32.129,
        'Vessel Material': 'stainless_steel',
        # Guard vessel intentionally removed: He is inert, no chemical-leak hazard requiring secondary containment
        'Gap Between Vessel And Guard Vessel': 0,
        'Guard Vessel Thickness': 0,  # cm
        'Guard Vessel Material': 'low_alloy_steel',
        'Gap Between Guard Vessel And Cooling Vessel': 5,  # cm
        'Cooling Vessel Thickness': 0.5,  # cm
        'Cooling Vessel Material': 'stainless_steel',
        'Gap Between Cooling Vessel And Intake Vessel': 5,  # cm — Hejzlar & Buongiorno 2007 NED RVACS minimum (was 4)
        'Intake Vessel Thickness': 0.5,  # cm
        'Intake Vessel Material': 'stainless_steel'
    })

    vessels_specs(params)
    calculate_shielding_masses(params)

    # **************************************************************************************************************************
    #                                           Sec. 9 : Operation
    # **************************************************************************************************************************
    update_params({
        'Operation Mode': "Remotely Monitored",
        'Number of Operators': 2,
        'Levelization Period': 60,  # years
        'Refueling Period': 7,
        'Emergency Shutdowns Per Year': 0.2,
        'Startup Duration after Refueling': 2,
        'Startup Duration after Emergency Shutdown': 14,
        'Reactors Monitored Per Operator': 10,
        'Security Staff Per Shift': 1
    })

    # Based on https://digital.library.unt.edu/ark:/67531/metadc893980/m2/1/high_res_d/919556.pdf (tables 17 and 18):
    # Estimated helium mass per MWt is 3.3 kg/MWt.
    params['Onsite Coolant Inventory'] = 3.3 * params['Power MWt']  # kg
    # According to https://www.nationalacademies.org/read/12844/chapter/6#69, the helium loss rate is 10% per year,
    # so 1/10 of the initial inventory is replenished annually.
    # Without purification, helium needs to be replaced more frequently.
    params['Replacement Coolant Inventory'] = params['Onsite Coolant Inventory'] / 10
    params['Annual Coolant Supply Frequency'] = 1 if params['Primary Loop Purification'] else 6

    total_refueling_period = params['Fuel Lifetime'] + params['Refueling Period'] + params['Startup Duration after Refueling'] # days
    total_refueling_period_yr = total_refueling_period/365
    params['A75: Vessel Replacement Period (cycles)']        = np.floor(10/total_refueling_period_yr)
    params['A75: Core Barrel Replacement Period (cycles)']   = np.floor(10/total_refueling_period_yr)
    params['A75: Reflector Replacement Period (cycles)']     = np.floor(10/total_refueling_period_yr)
    params['A75: Drum Replacement Period (cycles)']          = np.floor(10/total_refueling_period_yr)
    params['Maintenance to Direct Cost Ratio']                = 0.015
    params['A78: CAPEX to Decommissioning Cost Ratio'] = 0.15

    # **************************************************************************************************************************
    #                                           Sec. 10 : Economic Parameters
    # **************************************************************************************************************************
    update_params({
        'Land Area': 18,  # acres
        'Escalation Year': 2025,
        'Excavation Volume': 412.605,  # m^3
        'Reactor Building Slab Roof Volume': (9750*6502.4*1500)/1e9,  # m^3
        'Reactor Building Basement Volume': (9750*6502.4*1500)/1e9,  # m^3
        'Reactor Building Exterior Walls Volume': ((2*9750*3500*1500)+(3502.4*3500*(1500+750)))/1e9,  # m^3
        'Reactor Building Superstructure Area': ((2*3500*3500)+(2*7500*3500))/1e6, # m^2
        'Integrated Heat Exchanger Building Slab Roof Volume': 0,  # m^3
        'Integrated Heat Exchanger Building Basement Volume': 0,  # m^3
        'Integrated Heat Exchanger Building Exterior Walls Volume': 0,  # m^3
        'Integrated Heat Exchanger Building Superstructure Area': 0, # m^2
        'Turbine Building Slab Roof Volume': (12192*2438*200)/1e9,  # m^3
        'Turbine Building Basement Volume': (12192*2438*200)/1e9,  # m^3
        'Turbine Building Exterior Walls Volume': ((12192*2496*200)+(2038*2496*200))*2/1e9,  # m^3
        'Control Building Slab Roof Volume': (12192*2438*200)/1e9,  # m^3
        'Control Building Basement Volume': (12192*2438*200)/1e9,  # m^3
        'Control Building Exterior Walls Volume': ((12192*2496*200)+(2038*2496*200))*2/1e9,  # m^3
        'Manipulator Building Slab Roof Volume': (4876.8*2438.4*400)/1e9, # m^3
        'Manipulator Building Basement Volume': (4876.8*2438.4*1500)/1e9, # m^3
        'Manipulator Building Exterior Walls Volume': ((4876.8*4445*400)+(2038.4*4445*400*2))/1e9, # m^3
        'Refueling Building Slab Roof Volume': 0,  # m^3
        'Refueling Building Basement Volume': 0,  # m^3
        'Refueling Building Exterior Walls Volume': 0,  # m^3
        'Spent Fuel Building Slab Roof Volume': 0,  # m^3
        'Spent Fuel Building Basement Volume': 0,  # m^3
        'Spent Fuel Building Exterior Walls Volume': 0,  # m^3
        'Emergency Building Slab Roof Volume': 0,  # m^3
        'Emergency Building Basement Volume': 0,  # m^3
        'Emergency Building Exterior Walls Volume': 0,  # m^3
        'Storage Building Slab Roof Volume': (8400*3500*400)/1e9, # m^3
        'Storage Building Basement Volume': (8400*3500*400)/1e9, # m^3
        'Storage Building Exterior Walls Volume': ((8400*2700*400)+(3100*2700*400*2))/1e9, # m^3
        'Radwaste Building Slab Roof Volume': 0,  # m^3
        'Radwaste Building Basement Volume': 0,  # m^3
        'Radwaste Building Exterior Walls Volume': 0,  # m^3,
        'Interest Rate': 0.07,
        'Discount Rate': 0.07,
        'Construction Duration': 12,  # months
        'Debt To Equity Ratio': 1,
        'Annual Return': 0.0475,
        'NOAK Unit Number': 100,
    })

    # **************************************************************************************************************************
    #                                           Sec. 11: Post Processing
    # **************************************************************************************************************************
    params['Number of Samples'] = 100 # Accounting for cost uncertainties
    # Estimate costs using the cost database file and save the output to an Excel file
    cost_estimate = detailed_bottom_up_cost_estimate('cost/Cost_Database.xlsx', f"{output_dir}/output_GCMR_A_E5_{calc_id}.xlsx")
    elapsed_time = (time.time() - time_start) / 60  # Calculate execution time
    print('Execution time:', np.round(elapsed_time, 1), 'minutes')
    foak_col = next(c for c in cost_estimate.columns if c.startswith('FOAK Estimated Cost ('))
    noak_col = next(c for c in cost_estimate.columns if c.startswith('NOAK Estimated Cost ('))
    LCOE_estimate_FOAK = cost_estimate.loc[cost_estimate['Account'] == 'LCOE', foak_col].values[0]
    LCOE_estimate_NOAK = cost_estimate.loc[cost_estimate['Account'] == 'LCOE', noak_col].values[0]
    if LCOE_estimate_FOAK <= 0: # TODO: investigate how LCOE could become negative. for now, fix by disregarding these cases
        LCOE_estimate_FOAK = 1e10
    if LCOE_estimate_NOAK <= 0:
        LCOE_estimate_NOAK = 1e10
    results = (params['Temp Coeff 3D (2D corrected)'], params['Most Limiting Shutdown Margin 3D (2D corrected)'], params['Fuel Lifetime'], params['Max Peaking Factor'], LCOE_estimate_FOAK, LCOE_estimate_NOAK)
    return results
