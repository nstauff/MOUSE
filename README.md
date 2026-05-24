<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/logos/MOUSE-logo_R1_white.png">
    <source media="(prefers-color-scheme: light)" srcset="./assets/logos/MOUSE-logo_R1_black.png">
    <img alt="MOUSE logo" src="./assets/logos/MOUSE-logo_R1_black.png" width="500">
  </picture>
</p>

# Microreactor Optimization Using Simulation and Economics (MOUSE) 

## Web App

A hosted Streamlit version of MOUSE is available here:

**[Launch the MOUSE Streamlit App](https://mouse-microreactors.streamlit.app)**

The web app provides an interactive interface for early-stage microreactor scoping and cost estimation without requiring users to install OpenMC or WATTS locally.

## Motivation
The rising interest in nuclear microreactors has highlighted the need for comprehensive technoeconomic assessments. However, the scarcity of publicly available designs and cost data has posed significant challenges. The Microreactor Optimization Using Simulation and Economics (MOUSE) tool addresses this gap by integrating nuclear microreactor design with reactor economics.

## Description

MOUSE leverages the [OpenMC](https://github.com/openmc-dev/openmc) Monte Carlo Particle Transport Code to perform detailed core simulations for various microreactor designs, including Liquid-Metal Thermal Microreactors (LTMR), Gas-Cooled TRISO-Fueled Microreactors (GCMR), and Heat Pipe Microreactors (HPMR). It includes simplified calculations for balance of plant and operational performance. Economically, MOUSE provides bottom-up cost estimates covering preconstruction, direct, indirect, training, financial, O&M, and fuel costs. It calculates total capital costs and the levelized cost of energy (LCOE) for both first-of-a-kind and nth-of-a-kind microreactors using data from the MARVEL project and other literature.

## What MOUSE Does

MOUSE supports:
- Parametric microreactor design studies
- Bottom-up capital and operating cost estimation
- FOAK and NOAK LCOE calculations
- LTMR, GCMR, and HPMR reference designs
- Interactive web-based scoping through the Streamlit app

## MOUSE Tool Structure
<img src="./assets/mouse_diagram.png" />

### User-Defined Inputs
Users can modify design inputs or economic inputs such as:
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


## Ways to Use MOUSE

### Option 1: Use the Web App

Use the hosted Streamlit app for interactive microreactor scoping and cost estimation:

**[https://mouse-microreactors.streamlit.app](https://mouse-microreactors.streamlit.app)**

No local installation is required.

### Option 2: Run the Research Code Locally

For full OpenMC/WATTS-based workflows, install the required dependencies listed below.

## Prerequisites
Before running the full MOUSE research code locally, ensure that the following packages are installed:
- [OpenMC](https://github.com/openmc-dev/openmc)
- [WATTS](https://github.com/watts-dev/watts)

## Running Examples Locally

Users can specify reactor design inputs and/or economic parameters for the LTMR, GCMR, and HPMR examples in:
- `examples/watts_exec_LTMR.py`
- `examples/watts_exec_GCMR_Design_A.py`
- `examples/watts_exec_HPMR.py`

A complete detailed bottom-up cost estimation is obtained by running commands such as:
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

## Citation

If you use MOUSE in technical work, please cite the relevant reports and publications below.

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

## License

MOUSE is released under the MIT License. See [LICENSE](./LICENSE).
