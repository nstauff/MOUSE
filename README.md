<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/logos/MOUSE-logo_R1_white.png">
    <source media="(prefers-color-scheme: light)" srcset="./assets/logos/MOUSE-logo_R1_black.png">
    <img alt="MOUSE logo" src="./assets/logos/MOUSE-logo_R1_black.png" width="500">
  </picture>
</p>

# Microreactor Optimization Using Simulation and Economics (MOUSE) 
## Motivation
The rising interest in nuclear microreactors has highlighted the need for comprehensive technoeconomic assessments. However, the scarcity of publicly available designs and cost data has posed significant challenges. The Microreactor Optimization Using Simulation and Economics (MOUSE) tool addresses this gap by integrating nuclear microreactor design with reactor economics.

## Description

MOUSE leverages the [OpenMC](https://github.com/openmc-dev/openmc) Monte Carlo Particle Transport Code to perform detailed core simulations for various microreactor designs, including Liquid Metal Thermal Microreactors (LMTR), Gas-Cooled TRISO-Fueled Microreactors (GCMR), and Heat Pipe Microreactors. It includes simplified calculations for balance of plant and operational performance. Economically, MOUSE provides bottom-up cost estimates covering preconstruction, direct, indirect, training, financial, O&M, and fuel costs. It calculates total capital costs and the levelized cost of energy (LCOE) for both first-of-a-kind and nth-of-a-kind microreactors using data from the MARVEL project and other literature.

## MOUSE Tool Structure
<img src="./assets/mouse_diagram.png" />

### User-Defined Inputs
Users can modify design inputs or economic inputs ruch as:
- Overall System: Reactor Power (MWt), Thermal Efficiency (%), Heat Flux Criteria
- Geometry: Fuel Pin Radii, TRISO Packing Fraction, Coolant Channel Radius, Moderator Booster Radius, Lattice Pitch, Rings per Assembly, Assemblies per Core, Core Active Height, Reflector Thickness, Control Drum Dimensions
- Materials: Fuel, Enrichment, Coolant, Reflector, Matrix Material, Moderator, Moderator Booster, Control Drum Absorber/Reflector, Fuel Pin Materials
- Shielding: In/Out Vessel Shield Thickness, Material, Dimensions
- Vessels: Vessel Radius, Thickness, Materials, Gaps Between Vessels
- Balance of Plant: Coolant Inlet/Outlet Temperatures, Compressor Pressure Ratio, Pump Efficiency
- Operation: Operation Mode, Number of Operators, Plant Lifetime, Refueling Period, Number of Emergency Shutdowns per Year, Startup Durations
- Buildings: Dimensions of Reactor, Turbine, Control, Refueling, Spent Fuel, Emergency, Storage, Radioactive Waste Buildings
- Economic Parameters: Interest Rate, Dollar Escalation Year, Construction Duration, Debt to Equity Ratio

MOUSE is powered by the Workflow and Template Toolkit for Simulation ([WATTS](https://github.com/watts-dev/watts)), developed by ANL, which facilitates parametric studies by integrating various code components.

### Microreactor Designs
Three reactor designs are included so far:
- Liquid-metal thermal microreactor (LTMR)
- Gas-cooled microreactor (GCMR)
- Heat pipe microreactor (HPMR)

The designs can be found [here](./assets/Ref_openmc_2d_designs)


## Prerequisites
Before running the MOUSE code, ensure that the following packages are installed:
- [OpenMC](https://github.com/openmc-dev/openmc)
- [WATTS](https://github.com/watts-dev/watts)

### Getting started
The user can specifiy the reactor design specs and/or the economics parameters for the LTMR and GCMR in the files
`watts_exec_LTMR.py` or `watts_exec_GCMR_Design_A.py` or `watts_exec_HPMR.py`

A complete detailed bottom up cost estimation is obtained by running commands such as
```
python -m examples.watts_exec_LTMR
```
```
python -m examples.watts_exec_GCMR_Design_A
```
```
python -m examples.watts_exec_HPMR
```
Examples of the results are [here](./assets/Ref_Results)

## Relevant Publications

### Foundational Reports and Papers

**[Technoeconomic Evaluation of Microreactor Using Detailed Bottom-up Estimate (Rev. 1)](https://www.osti.gov/biblio/2447366)**
Hanna et al., INL Technical Report, 2024. Develops the transparent bottom-up cost methodology using the MARVEL microreactor as reference — the direct precursor to MOUSE's economic framework.

---

**[A Bottom-Up Cost Estimation Tool for Nuclear Microreactors](https://www.osti.gov/biblio/2588465)**
Hanna et al., INL S&T Accomplishment Report, 2025. Describes the MOUSE tool, its architecture, and demonstrates FOAK/NOAK cost estimates and parametric studies for the LTMR, GCMR, and HPMR designs.

---

**[Open-Source Microreactor Design Models for Technoeconomic Assessments](https://www.sciencedirect.com/science/article/pii/S0029549325003875)**
Al-Dawood, Hanna et al., Nuclear Engineering and Design, 2025. Documents the cost correlations, design parameters, and bottom-up assumptions for the LTMR and GCMR models embedded in MOUSE.

---

### Applications and Extensions

**[Techno-Economic Optimization of a Heat-Pipe Microreactor, Part I: Theory and Cost Optimization](https://arxiv.org/pdf/2512.16032)**
Seurin, Price, Nunez — INL/MIT, 2025. Couples MOUSE's LCOE estimation with surrogate modeling and reinforcement learning to optimize HPMR geometry under physics constraints.

---

**[Techno-Economic Optimization of a Heat-Pipe Microreactor, Part II: Multi-Objective Optimization Analysis](https://arxiv.org/pdf/2601.20079)**
Seurin, Price — INL/MIT, 2026. Extends Part I to multi-objective optimization of LCOE and rod-integrated peaking factor using the PEARL algorithm, with MOUSE as the cost engine.