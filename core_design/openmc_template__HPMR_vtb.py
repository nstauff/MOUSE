import openmc
import pandas as pd
import random
import openmc.deplete
import itertools
import numpy as np
from pathlib import Path
import shutil

def drum_parameters_to_eq(radius, coating_angle, drum_angle, p0):
    """convert control drum parameters to A, B, C & D intersecting plane parameters"""
    coating_angle = coating_angle*np.pi/180
    drum_angle = drum_angle*np.pi/180
    l = radius*np.cos(coating_angle / 2)
    point = np.array([[np.cos(drum_angle), -np.sin(drum_angle)],[np.sin(drum_angle), np.cos(drum_angle)]])@np.array([l, 0]) + p0
    norm = np.array([np.cos(drum_angle), np.sin(drum_angle)])
    A, B = norm
    C = 0
    D = A*point[0] + B*point[1]
    if p0[0] > 0 or (np.isclose(p0[0], 0) and p0[1] > 0): #takes care of positivity of plane normal
        return A,B,C,D
    else:
        return -A,-B,-C,-D

def clear_dir(rundir):
    if rundir.exists() and rundir.is_dir():
        for item in rundir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

def write_dict(data_dict, file_path):
    with open(file_path, 'w') as file:
        file.write('Parameter,Value\n')
        for key, value in data_dict.items():
            file.write(f'{key},{value}\n')

def find_crossing_time(time, k):
    if k[0] < 1:
        t1, t2 = time[1], time[2]
        k1, k2 = k[1], k[2]
        crossing_time = t1 - (1 - k1) * (t2 - t1) / (k2 - k1)
    if np.any(k <= 1): #is k < 1 by end of sim?
        idx = np.where(k <= 1)[0][0]
        t1, t2 = time[idx-1], time[idx]
        k1, k2 = k[idx-1], k[idx]
        crossing_time = t1 + (1 - k1) * (t2 - t1) / (k2 - k1)
    else:
        t1, t2 = time[-2], time[-1]
        k1, k2 = k[-2], k[-1]
        crossing_time = t2 + (1 - k2) * (t2 - t1) / (k2 - k1)
    return crossing_time


def pin_centers(pitch, flake_width):
    centers = []
    rows = 13
    cols = 13
    for row in range(rows):
        for col in range(cols):
            x = pitch/np.sqrt(3) * col * 3/2
            y = pitch/np.sqrt(3) * (row * np.sqrt(3) + (col % 2) * np.sqrt(3) / 2)
            centers.append((x - 3*pitch*np.sqrt(3), y - 6*pitch))
    centers = np.array(centers)
    msk = np.array([52,40,27,78,66,67,55,42,29,16,104,92,93,81,82,70,57,44,31,18,5,37,50,64,90,75,61,48,35,22,116,102,101,86,87,76,72,59,46,33,20,7,148,149,150,133,134,122,123,124,138,139,118,119,107,108,96,97,98,112,113,127,128])
    centers = centers[msk]
    return centers
    
def flake_centers(flake_width):
    centers = []
    rows = 8
    cols = 8
    for row in range(rows):
        for col in range(cols):
            y = flake_width/np.sqrt(3) * col * 3/2
            x = flake_width/np.sqrt(3) * (row * np.sqrt(3) + (col % 2) * np.sqrt(3) / 2)
            centers.append((x - 7/2*flake_width , y - 3/2*flake_width*np.sqrt(3)))
    centers = np.array(centers)
    msk = [24,32,9,17,25,33,41,10,18,26,34,42,50,11,19,35,43,12,20,28,36,44,52,13,21,29,37,45,30,38]
    centers = centers[msk]
    return centers

def all_centers(pitch, flake_width):
    centers = pd.DataFrame(np.zeros((1890, 3)), columns = ["flake_id", "x", "y"])
    p_cents_base = pin_centers(pitch, flake_width)
    flake_cents = flake_centers(flake_width)
    p_cents = []
    for fc in flake_cents:
        for p_cent_base in p_cents_base:
            p_cents.append([fc[0]+p_cent_base[0], fc[1]+p_cent_base[1]])
    p_cents = np.array(p_cents)
    return p_cents

def group_powers(x_centers, y_centers, ta, pin_centers):
    # Calculate the map
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
    mp = np.zeros((x_grid.shape[0], y_grid.shape[1]), dtype=int)

    zsum = ta.sum(2)
    for i in range(x_grid.shape[0]):
        for j in range(y_grid.shape[1]):
            if zsum[i,j] > 0:
                distances = np.sum((pin_centers - np.array([x_grid[i, j], y_grid[i, j]]))**2, axis=1)
                mp[i, j] = np.argmin(distances) + 1 #offset for placeholder
            else:
                mp[i,j] = 0 #placeholder

    # Initialize the powers dictionary
    powers = {tuple(pin_centers[i]): np.zeros(ta.shape[2]) for i in range(pin_centers.shape[0])}

    # Sum the values in ta according to the mp
    for i in range(mp.shape[0]):
        for j in range(mp.shape[1]):
            if mp[i,j] != 0:
                pin_index = mp[i, j] - 1
                powers[tuple(pin_centers[pin_index])] += ta[i, j, :]
    return powers


def proc_pinpower(ta, ll, ur, pitch, flake_width):
    #define voxel centers
    shp = ta.shape
    x_centers = np.linspace(ll[0] + (ur[0] - ll[0]) / (2 * shp[0]), ur[0] - (ur[0] - ll[0]) / (2 * shp[0]), shp[0])
    y_centers = np.linspace(ll[1] + (ur[1] - ll[1]) / (2 * shp[1]), ur[1] - (ur[1] - ll[1]) / (2 * shp[1]), shp[1])
    z_centers = np.linspace(ll[2] + (ur[2] - ll[2]) / (2 * shp[2]), ur[2] - (ur[2] - ll[2]) / (2 * shp[2]), shp[2])

    #assign coordinates to indices and find centers
    pin_centers = all_centers(pitch, flake_width)
    pin_powers = group_powers(x_centers, y_centers, ta, pin_centers)
    return pin_powers, z_centers

def write_pinpowers(working_dir, pin_powers, z_coords):
    f = open(working_dir / Path("run_90_pinpowers.csv"), "w")
    f.write("z coord centers:")
    f.write(",".join([str(a) for a in z_coords]))
    f.write("\ncenter: powers\n")
    for k, v in pin_powers.items():
        f.write(f"({str(k[0])},{str(k[1])}): ") 
        for e in v:
            f.write(str(e) + ",")
        f.write("\n")
    f.close()

def calculate_average_power_transfer(Power,N_flakes,N_compact):
    # Functions to calculate average power transfer by a compact
    return Power / N_flakes / N_compact

def calculate_average_q_prime(Power,N_flakes,N_compact,active_fuel_height):
    # Functions to calculate average linear heat generation rate
    return calculate_average_power_transfer(Power,N_flakes,N_compact) / active_fuel_height


def calculate_heat_flux(Power,N_flakes,N_compact,compact_radius,active_height):
    # Functions to calculate average heat flux 
    # assume tallies are such that the compact is a cylindrical energy source 
    fuel_number =  N_flakes*N_compact
    heat_transfer_surface = cylinder_radial_shell(compact_radius, active_height ) * fuel_number  * 1e-4 # convert from cm2 to m2
    
    return Power/heat_transfer_surface # MW/m^2

def circle_perimeter(r):
    return 2*(np.pi)*r

def cylinder_radial_shell(r, h):
    # calculating the outer area of a cylinder
    return circle_perimeter(r) * h

class OpenMC_HPMR:
    """Class works on parameter dictionaries, see "self.nomoinal_params" for example"""
    default_particles = 100000
    default_batches = 500
    default_inactive = 50
    default_low_pf = False
    def __init__(self, working_dir = Path("."), chain_file = None):
        self.chain_file = chain_file  # depletion chain XML; required for deplete()
        self.nominal_parms = {
                "compact_radius" : 1.00, #cm
                "moderator_radius" : 0.825, #cm; radius of moderator material in mod pin, cladding thickness stays const
                "coating_angle" : 90., #deg; control drum coating angle
                "B10_at_frac_B" : .95, #B10/(B10 + B11) as atom fraction
                "flake_width" : 26.752, #cm; flat to flat width of flakes, control drum scaled according to r=(w-0.252)/2. Inner radius scaled too
                "pin_pitch" : 2.3, #cm; width of pin hex cell
                "enrichment":0.197, #should be between 0 and 1
                "reflector_width" : 260., #cm; flat to flat width of radial reflector
                "active_fuel_height" : 160., #cm
                "axial_reflector_height" : 20. #cm
                }
        self.working_dir = working_dir

        self.Tvalues = {"DrumRot" : [0, 90, 180]}
        self.Tcombos = [dict(zip(self.Tvalues.keys(), combo)) for combo in itertools.product(*self.Tvalues.values())]

    def run_nominal(self,
            threads = 1, #openmc threads
            particles = default_particles, batches = default_batches, inactive = default_inactive, #MC sampling parameters
            plot = True, #whether or not to generate plots
            axial_divs = 11,
            low_pf = default_low_pf): #debugging option for faster particle dispersions
        """generate case matrix for nominal model"""
        self.run_perturbed(self.nominal_parms, threads, particles, batches, inactive, plot, axial_divs, low_pf)

    def run_perturbed(self,
            perturbed_parms, #dict with parameter names and values
            threads = 1, #openmc threads
            particles = default_particles, batches = default_batches, inactive = default_inactive, #MC sampling parameters
            plot = True,
            axial_divs = 11,
            low_pf = default_low_pf):
        write_dict(perturbed_parms, self.working_dir / Path("design_parameters.txt"))
        """generate case matrix for perturbed model"""
        for Tdict in self.Tcombos:
            Drmidx = self.Tvalues["DrumRot"].index(Tdict["DrumRot"]) + 1
            rundir = self.working_dir / Path(f"run_{Tdict['DrumRot']}")
            rundir.mkdir(parents = True, exist_ok = True)
            clear_dir(rundir)
            if Tdict["DrumRot"] == 0:
                curr_model, fuel_mats = self.gen_input(perturbed_parms, Tdict, rundir, particles, batches, inactive, axial_divs, tally = False, low_pf = low_pf)
                openmc.run(threads = threads, cwd = rundir.resolve(), path_input = rundir.resolve())
            elif Tdict["DrumRot"] == 90:
                print("Running double particles for peaking factors")
                curr_model, fuel_mats = self.gen_input(perturbed_parms, Tdict, rundir, 2*particles, 2*batches, inactive, axial_divs, tally = True, low_pf = low_pf)
                openmc.run(threads = threads, cwd = rundir.resolve(), path_input = rundir.resolve())

                # run hot zero power case
                # assume operating temperature of the sodium in the heat pipe is 
                rundir_hzp = self.working_dir / Path(f"run_hzp_{Tdict['DrumRot']}")
                rundir_hzp.mkdir(parents = True, exist_ok = True)
                clear_dir(rundir_hzp)
                curr_model, fuel_mats = self.gen_input(perturbed_parms, Tdict, rundir_hzp, particles, batches, inactive, axial_divs, tally = False, low_pf = low_pf,operating_temperature=800)
                openmc.run(threads = threads, cwd = rundir_hzp.resolve(), path_input = rundir_hzp.resolve())
                # run most effective drums in. Any in this case.
                rundir_meri = self.working_dir / Path(f"run_meri_{Tdict['DrumRot']}")
                rundir_meri.mkdir(parents = True, exist_ok = True)
                clear_dir(rundir_meri)
                curr_model, fuel_mats = self.gen_input(perturbed_parms, Tdict, rundir_meri, particles, batches, inactive, axial_divs, tally = False, low_pf = low_pf,most_effective_rods_in=True)
                openmc.run(threads = threads, cwd = rundir_meri.resolve(), path_input = rundir_meri.resolve())
                if plot:
                    openmc.plot_geometry(path_input = rundir_hzp.resolve(), cwd = rundir_hzp.resolve())
                    openmc.plot_geometry(path_input = rundir_meri.resolve(), cwd = rundir_meri.resolve())
            elif Tdict["DrumRot"] == 180:
                curr_model, fuel_mats = self.gen_input(perturbed_parms, Tdict, rundir, particles, batches, inactive, axial_divs, tally = False, low_pf = low_pf)
                if low_pf:
                    deplete = [0.1] + [3]*4
                else:
                    deplete = [0.1] + [740]*4
                self.deplete(rundir = rundir.resolve(), model = curr_model, fuel_mats = fuel_mats, deplete = deplete)
            if plot:
                openmc.plot_geometry(path_input = rundir.resolve(), cwd = rundir.resolve())
            

    def deplete(self, rundir, model, fuel_mats, deplete):
        if self.chain_file is None:
            raise ValueError(
                "OpenMC_HPMR.chain_file is not set. Pass chain_file=... to the "
                "constructor, e.g. OpenMC_HPMR(chain_file=params['simplified_chain_thermal_xml'])."
            )
        operator = openmc.deplete.CoupledOperator(model, chain_file = self.chain_file)
        pi = openmc.deplete.PredictorIntegrator(timestep_units = "d",
                timesteps = deplete,
                operator = operator,
                power = 2e6)
        pi.integrate(path = rundir / Path("depletion_results.h5"))
        for f in Path(".").iterdir():
            if f.suffix in [".h5", ".xml"]:
                shutil.move(str(f), str(Path("run_180") / f.name))

    def gen_input(self, parms, Tdict, rundir, particles, batches, inactive, axial_divs, tally, low_pf, operating_temperature = 850,most_effective_rods_in=False):
        all_mats = openmc.Materials()
        
        air = openmc.Material(name='air')# -0.18e-3   rgb 255 255 255   tmp  operating_temperature
        air.add_nuclide('He4', 1.0, percent_type='ao')
        air.set_density('g/cm3', 0.18e-3)
        air.temperature = operating_temperature
        all_mats.append(air)
        
        shell_mod = openmc.Material(name='shell_mod') # -7.90   rgb 133 133 133    tmp  operating_temperature
        shell_mod.add_nuclide('C12',1.9010E-03, percent_type='ao')
        shell_mod.add_nuclide('Si28',  9.2693E-03, percent_type='ao')
        shell_mod.add_nuclide('Si29',  4.7251E-04, percent_type='ao')
        shell_mod.add_nuclide('Si30',  3.1166E-04, percent_type='ao')
        shell_mod.add_nuclide('P31',  4.1322E-04, percent_type='ao')
        shell_mod.add_nuclide('S32',  2.4710E-04, percent_type='ao')
        shell_mod.add_nuclide('S33',  1.9511E-06, percent_type='ao')
        shell_mod.add_nuclide('S34',  1.1056E-05, percent_type='ao')
        shell_mod.add_nuclide('S36',  3.0016E-08, percent_type='ao')
        shell_mod.add_nuclide('Cr50',  7.9116E-03, percent_type='ao')
        shell_mod.add_nuclide('Cr52',  1.5257E-01, percent_type='ao')
        shell_mod.add_nuclide('Cr53',  1.7300E-02, percent_type='ao')
        shell_mod.add_nuclide('Cr54',  4.3063E-03, percent_type='ao')
        shell_mod.add_nuclide('Mn55',  1.0280E-02, percent_type='ao')
        shell_mod.add_nuclide('Fe54',  3.9029E-02, percent_type='ao')
        shell_mod.add_nuclide('Fe56',  6.1213E-01, percent_type='ao')
        shell_mod.add_nuclide('Fe57',  1.4144E-02, percent_type='ao')
        shell_mod.add_nuclide('Fe58',  1.3343E-03, percent_type='ao')
        shell_mod.add_nuclide('Ni58',  7.7516E-02, percent_type='ao')
        shell_mod.add_nuclide('Ni60',  2.9859E-02, percent_type='ao')
        shell_mod.add_nuclide('Ni61',  1.2981E-03, percent_type='ao')
        shell_mod.add_nuclide('Ni62',  4.1389E-03, percent_type='ao')
        shell_mod.add_nuclide('Ni64',  1.0544E-03, percent_type='ao')
        shell_mod.add_nuclide('Mo92',  2.1259E-03, percent_type='ao')
        shell_mod.add_nuclide('Mo94',  1.3299E-03, percent_type='ao')
        shell_mod.add_nuclide('Mo95',  2.3030E-03, percent_type='ao')
        shell_mod.add_nuclide('Mo96',  2.4191E-03, percent_type='ao')
        shell_mod.add_nuclide('Mo97',  1.3903E-03, percent_type='ao')
        shell_mod.add_nuclide('Mo98',  3.5249E-03, percent_type='ao')
        shell_mod.add_nuclide('Mo100',  1.4135E-03, percent_type='ao')
        shell_mod.set_density('g/cm3', 7.90)
        shell_mod.temperature = operating_temperature
        all_mats.append(shell_mod)
        
        shell_hp = openmc.Material(name='shell_hp') #mat shell_hp -7.90   rgb 133 133 133    tmp  operating_temperature
        shell_hp.add_nuclide('C12',   1.9010E-03, percent_type='ao')
        shell_hp.add_nuclide('Si28',  9.2693E-03, percent_type='ao')
        shell_hp.add_nuclide('Si29',  4.7251E-04, percent_type='ao')
        shell_hp.add_nuclide('Si30',  3.1166E-04, percent_type='ao')
        shell_hp.add_nuclide('P31',   4.1322E-04, percent_type='ao')
        shell_hp.add_nuclide('S32',   2.4710E-04, percent_type='ao')
        shell_hp.add_nuclide('S33',   1.9511E-06, percent_type='ao')
        shell_hp.add_nuclide('S34',   1.1056E-05, percent_type='ao')
        shell_hp.add_nuclide('S36',   3.0016E-08, percent_type='ao')
        shell_hp.add_nuclide('Cr50',  7.9116E-03, percent_type='ao')
        shell_hp.add_nuclide('Cr52',  1.5257E-01, percent_type='ao')
        shell_hp.add_nuclide('Cr53',  1.7300E-02, percent_type='ao')
        shell_hp.add_nuclide('Cr54',  4.3063E-03, percent_type='ao')
        shell_hp.add_nuclide('Mn55',  1.0280E-02, percent_type='ao')
        shell_hp.add_nuclide('Fe54',  3.9029E-02, percent_type='ao')
        shell_hp.add_nuclide('Fe56',  6.1213E-01, percent_type='ao')
        shell_hp.add_nuclide('Fe57',  1.4144E-02, percent_type='ao')
        shell_hp.add_nuclide('Fe58',  1.3343E-03, percent_type='ao')
        shell_hp.add_nuclide('Ni58',  7.7516E-02, percent_type='ao')
        shell_hp.add_nuclide('Ni60',  2.9859E-02, percent_type='ao')
        shell_hp.add_nuclide('Ni61',  1.2981E-03, percent_type='ao')
        shell_hp.add_nuclide('Ni62',  4.1389E-03, percent_type='ao')
        shell_hp.add_nuclide('Ni64',  1.0544E-03, percent_type='ao')
        shell_hp.add_nuclide('Mo92',  2.1259E-03, percent_type='ao')
        shell_hp.add_nuclide('Mo94',  1.3299E-03, percent_type='ao')
        shell_hp.add_nuclide('Mo95',  2.3030E-03, percent_type='ao')
        shell_hp.add_nuclide('Mo96',  2.4191E-03, percent_type='ao')
        shell_hp.add_nuclide('Mo97',  1.3903E-03, percent_type='ao')
        shell_hp.add_nuclide('Mo98',  3.5249E-03, percent_type='ao')
        shell_hp.add_nuclide('Mo100', 1.4135E-03, percent_type='ao')
        shell_hp.set_density('g/cm3', 7.90)
        shell_hp.temperature = operating_temperature
        all_mats.append(shell_hp)
        
        shell_air_mod = openmc.Material(name='shell_air_mod') #mat shell_air_mod 2.290E-02 rgb 133 133 133    tmp  operating_temperature
        shell_air_mod.add_nuclide('He4',    1.983E-05, percent_type='ao')
        shell_air_mod.add_nuclide('C12',    4.349E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Si28',   2.121E-04, percent_type='ao')
        shell_air_mod.add_nuclide('Si29',   1.081E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Si30',   7.130E-06, percent_type='ao')
        shell_air_mod.add_nuclide('P31',    9.454E-06, percent_type='ao')
        shell_air_mod.add_nuclide('S32',    5.653E-06, percent_type='ao')
        shell_air_mod.add_nuclide('S33',    4.464E-08, percent_type='ao')
        shell_air_mod.add_nuclide('S34',    2.530E-07, percent_type='ao')
        shell_air_mod.add_nuclide('S36',    6.867E-10, percent_type='ao')
        shell_air_mod.add_nuclide('Cr50',   1.810E-04, percent_type='ao')
        shell_air_mod.add_nuclide('Cr52',   3.491E-03, percent_type='ao')
        shell_air_mod.add_nuclide('Cr53',   3.958E-04, percent_type='ao')
        shell_air_mod.add_nuclide('Cr54',   9.852E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Mn55',   2.352E-04, percent_type='ao')
        shell_air_mod.add_nuclide('Fe54',   8.929E-04, percent_type='ao')
        shell_air_mod.add_nuclide('Fe56',   1.400E-02, percent_type='ao')
        shell_air_mod.add_nuclide('Fe57',   3.236E-04, percent_type='ao')
        shell_air_mod.add_nuclide('Fe58',   3.053E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Ni58',   1.773E-03, percent_type='ao')
        shell_air_mod.add_nuclide('Ni60',   6.831E-04, percent_type='ao')
        shell_air_mod.add_nuclide('Ni61',   2.970E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Ni62',   9.469E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Ni64',   2.412E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Mo92',   4.864E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Mo94',   3.043E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Mo95',   5.269E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Mo96',   5.535E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Mo97',   3.181E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Mo98',   8.065E-05, percent_type='ao')
        shell_air_mod.add_nuclide('Mo100',  3.234E-05, percent_type='ao')
        shell_air_mod.set_density('atom/b-cm', 2.290E-02)
        shell_air_mod.temperature = operating_temperature
        all_mats.append(shell_air_mod)
        
        shell_air_hp = openmc.Material(name='shell_air_hp') #mat shell_air_hp 6.771E-02 rgb 133 133 133    tmp  operating_temperature
        shell_air_hp.add_nuclide('He4',   5.629E-06, percent_type='ao')
        shell_air_hp.add_nuclide('C12',   1.287E-04, percent_type='ao')
        shell_air_hp.add_nuclide('Si28',  6.276E-04, percent_type='ao')
        shell_air_hp.add_nuclide('Si29',  3.199E-05, percent_type='ao')
        shell_air_hp.add_nuclide('Si30',  2.110E-05, percent_type='ao')
        shell_air_hp.add_nuclide('P31',   2.798E-05, percent_type='ao')
        shell_air_hp.add_nuclide('S32',   1.673E-05, percent_type='ao')
        shell_air_hp.add_nuclide('S33',   1.321E-07, percent_type='ao')
        shell_air_hp.add_nuclide('S34',   7.486E-07, percent_type='ao')
        shell_air_hp.add_nuclide('S36',   2.032E-09, percent_type='ao')
        shell_air_hp.add_nuclide('Cr50',  5.357E-04, percent_type='ao')
        shell_air_hp.add_nuclide('Cr52',  1.033E-02, percent_type='ao')
        shell_air_hp.add_nuclide('Cr53',  1.171E-03, percent_type='ao')
        shell_air_hp.add_nuclide('Cr54',  2.916E-04, percent_type='ao')
        shell_air_hp.add_nuclide('Mn55',  6.960E-04, percent_type='ao')
        shell_air_hp.add_nuclide('Fe54',  2.643E-03, percent_type='ao')
        shell_air_hp.add_nuclide('Fe56',  4.145E-02, percent_type='ao')
        shell_air_hp.add_nuclide('Fe57',  9.576E-04, percent_type='ao')
        shell_air_hp.add_nuclide('Fe58',  9.034E-05, percent_type='ao')
        shell_air_hp.add_nuclide('Ni58',  5.248E-03, percent_type='ao')
        shell_air_hp.add_nuclide('Ni60',  2.022E-03, percent_type='ao')
        shell_air_hp.add_nuclide('Ni61',  8.789E-05, percent_type='ao')
        shell_air_hp.add_nuclide('Ni62',  2.802E-04, percent_type='ao')
        shell_air_hp.add_nuclide('Ni64',  7.139E-05, percent_type='ao')
        shell_air_hp.add_nuclide('Mo92',  1.439E-04, percent_type='ao')
        shell_air_hp.add_nuclide('Mo94',  9.004E-05, percent_type='ao')
        shell_air_hp.add_nuclide('Mo95',  1.559E-04, percent_type='ao')
        shell_air_hp.add_nuclide('Mo96',  1.638E-04, percent_type='ao')
        shell_air_hp.add_nuclide('Mo97',  9.413E-05, percent_type='ao')
        shell_air_hp.add_nuclide('Mo98',  2.387E-04, percent_type='ao')
        shell_air_hp.add_nuclide('Mo100', 9.570E-05, percent_type='ao')
        shell_air_hp.set_density('atom/b-cm', 6.771E-02)
        shell_air_hp.temperature = operating_temperature
        all_mats.append(shell_air_hp)
        
        shell_air_center = openmc.Material(name='shell_air_center') #mat shell_air_center 8.815E-04 rgb 133 133 133  tmp  operating_temperature
        shell_air_center.add_nuclide('He4',    2.68E-05, percent_type='ao')
        shell_air_center.add_nuclide('C12',   1.62E-06, percent_type='ao')
        shell_air_center.add_nuclide('Si28',   7.92E-06, percent_type='ao')
        shell_air_center.add_nuclide('Si29',   4.04E-07, percent_type='ao')
        shell_air_center.add_nuclide('Si30',   2.66E-07, percent_type='ao')
        shell_air_center.add_nuclide('P31',    3.53E-07, percent_type='ao')
        shell_air_center.add_nuclide('S32',    2.11E-07, percent_type='ao')
        shell_air_center.add_nuclide('S33',    1.67E-09, percent_type='ao')
        shell_air_center.add_nuclide('S34',    9.45E-09, percent_type='ao')
        shell_air_center.add_nuclide('S36',    2.57E-11, percent_type='ao')
        shell_air_center.add_nuclide('Cr50',   6.76E-06, percent_type='ao')
        shell_air_center.add_nuclide('Cr52',   1.30E-04, percent_type='ao')
        shell_air_center.add_nuclide('Cr53',   1.48E-05, percent_type='ao')
        shell_air_center.add_nuclide('Cr54',   3.68E-06, percent_type='ao')
        shell_air_center.add_nuclide('Mn55',   8.79E-06, percent_type='ao')
        shell_air_center.add_nuclide('Fe54',   3.34E-05, percent_type='ao')
        shell_air_center.add_nuclide('Fe56',   5.23E-04, percent_type='ao')
        shell_air_center.add_nuclide('Fe57',   1.21E-05, percent_type='ao')
        shell_air_center.add_nuclide('Fe58',   1.14E-06, percent_type='ao')
        shell_air_center.add_nuclide('Ni58',   6.63E-05, percent_type='ao')
        shell_air_center.add_nuclide('Ni60',   2.55E-05, percent_type='ao')
        shell_air_center.add_nuclide('Ni61',   1.11E-06, percent_type='ao')
        shell_air_center.add_nuclide('Ni62',   3.54E-06, percent_type='ao')
        shell_air_center.add_nuclide('Ni64',   9.01E-07, percent_type='ao')
        shell_air_center.add_nuclide('Mo92',   1.82E-06, percent_type='ao')
        shell_air_center.add_nuclide('Mo94',   1.14E-06, percent_type='ao')
        shell_air_center.add_nuclide('Mo95',   1.97E-06, percent_type='ao')
        shell_air_center.add_nuclide('Mo96',   2.07E-06, percent_type='ao')
        shell_air_center.add_nuclide('Mo97',   1.19E-06, percent_type='ao')
        shell_air_center.add_nuclide('Mo98',   3.01E-06, percent_type='ao')
        shell_air_center.add_nuclide('Mo100',  1.21E-06, percent_type='ao')
        shell_air_center.set_density('atom/b-cm', 8.815E-04)
        shell_air_center.temperature = operating_temperature
        all_mats.append(shell_air_center)
        
        coo_vap = openmc.Material(name='coo_vap') #mat coo_vap  -1.11e-4 rgb 250 50 250    tmp  operating_temperature
        coo_vap.add_nuclide('K39',   0.93258, percent_type='ao')
        coo_vap.add_nuclide('K40',   0.00012, percent_type='ao')
        coo_vap.add_nuclide('K41',   0.06730, percent_type='ao')
        coo_vap.set_density('g/cm3', 1.11e-4)
        coo_vap.temperature = operating_temperature
        all_mats.append(coo_vap)
        
        coo_liq = openmc.Material(name='coo_liq') #mat coo_liq  -0.705 rgb 50 250 50    tmp  operating_temperature
        coo_liq.add_nuclide('K39',   0.93258, percent_type='ao')
        coo_liq.add_nuclide('K40',   0.00012, percent_type='ao')
        coo_liq.add_nuclide('K41',   0.06730, percent_type='ao')
        coo_liq.set_density('g/cm3', 0.705)
        coo_liq.temperature = operating_temperature
        all_mats.append(coo_liq)
        
        wick = openmc.Material(name='wick')# mat wick  -2.753 rgb 250 250 50     tmp  operating_temperature
        wick.add_nuclide('C12',   5.589E-04, percent_type='ao')
        wick.add_nuclide('Si28',  2.725E-03, percent_type='ao')
        wick.add_nuclide('Si29',  1.389E-04, percent_type='ao')
        wick.add_nuclide('Si30',  9.163E-05, percent_type='ao')
        wick.add_nuclide('P31',   1.215E-04, percent_type='ao')
        wick.add_nuclide('S32',   7.265E-05, percent_type='ao')
        wick.add_nuclide('S33',   5.736E-07, percent_type='ao')
        wick.add_nuclide('S34',   3.250E-06, percent_type='ao')
        wick.add_nuclide('S36',   8.825E-09, percent_type='ao')
        wick.add_nuclide('Cr50',  2.326E-03, percent_type='ao')
        wick.add_nuclide('Cr52',  4.485E-02, percent_type='ao')
        wick.add_nuclide('Cr53',  5.086E-03, percent_type='ao')
        wick.add_nuclide('Cr54',  1.266E-03, percent_type='ao')
        wick.add_nuclide('Mn55',  3.022E-03, percent_type='ao')
        wick.add_nuclide('Fe54',  1.147E-02, percent_type='ao')
        wick.add_nuclide('Fe56',  1.800E-01, percent_type='ao')
        wick.add_nuclide('Fe57',  4.158E-03, percent_type='ao')
        wick.add_nuclide('Fe58',  3.923E-04, percent_type='ao')
        wick.add_nuclide('Ni58',  2.279E-02, percent_type='ao')
        wick.add_nuclide('Ni60',  8.778E-03, percent_type='ao')
        wick.add_nuclide('Ni61',  3.816E-04, percent_type='ao')
        wick.add_nuclide('Ni62',  1.217E-03, percent_type='ao')
        wick.add_nuclide('Ni64',  3.100E-04, percent_type='ao')
        wick.add_nuclide('Mo92',  6.250E-04, percent_type='ao')
        wick.add_nuclide('Mo94',  3.910E-04, percent_type='ao')
        wick.add_nuclide('Mo95',  6.771E-04, percent_type='ao')
        wick.add_nuclide('Mo96',  7.112E-04, percent_type='ao')
        wick.add_nuclide('Mo97',  4.087E-04, percent_type='ao')
        wick.add_nuclide('Mo98',  1.036E-03, percent_type='ao')
        wick.add_nuclide('Mo100', 4.156E-04, percent_type='ao')
        wick.add_nuclide('K39',   6.584E-01, percent_type='ao')
        wick.add_nuclide('K40',   8.472E-05, percent_type='ao')
        wick.add_nuclide('K41',   4.751E-02, percent_type='ao')
        wick.set_density('g/cm3', 2.753)
        wick.temperature = operating_temperature
        all_mats.append(wick)
        
        hp_vp_liq_wick = openmc.Material(name='hp_vp_liq_wick')# mat hp_vp_liq_wick  8.324E-03  rgb 250 250 50     tmp  operating_temperature
        hp_vp_liq_wick.add_nuclide('C12',   3.808E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Si28',   1.856E-05, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Si29',   9.463E-07, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Si30',   6.242E-07, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('P31',    8.277E-07, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('S32',    4.949E-07, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('S33',    3.908E-09, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('S34',    2.214E-08, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('S36',    6.012E-11, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Cr50',   1.585E-05, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Cr52',   3.055E-04, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Cr53',   3.465E-05, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Cr54',   8.625E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Mn55',   2.059E-05, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Fe54',   7.814E-05, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Fe56',   1.226E-03, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Fe57',   2.833E-05, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Fe58',   2.673E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Ni58',   1.553E-04, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Ni60',   5.980E-05, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Ni61',   2.600E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Ni62',   8.291E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Ni64',   2.112E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Mo92',   4.258E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Mo94',   2.664E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Mo95',   4.613E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Mo96',   4.845E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Mo97',   2.784E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Mo98',   7.058E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('Mo100',  2.831E-06, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('K39',    5.895E-03, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('K40',    7.586E-07, percent_type='ao')
        hp_vp_liq_wick.add_nuclide('K41',    4.254E-04, percent_type='ao')
        hp_vp_liq_wick.set_density('atom/b-cm', 8.324E-03)
        hp_vp_liq_wick.temperature = operating_temperature
        all_mats.append(hp_vp_liq_wick)
        
        #therm h-yh2 h-yh2.84t
        #therm y-yh2 y-yh2.84t
        moderator = openmc.Material(name='moderator') #mat moderator   -4.0850  moder h-yh2 1001 moder y-yh2 39089  tmp  operating_temperature.0   rgb 73 64 171
        moderator.add_nuclide('Y89',  0.357142857, percent_type='ao')
        moderator.add_nuclide('H1',   0.642857143, percent_type='ao')
        moderator.add_s_alpha_beta('c_H_in_YH2')
        moderator.add_s_alpha_beta('c_Y_in_YH2')
        moderator.set_density('g/cm3', 4.0850)
        moderator.temperature = operating_temperature
        all_mats.append(moderator)
        
        #% -- Fuel
        
       
        # To comply with MOUSE framework
        #
        #UO2
        UO2 = openmc.Material(name='UO2')
        UO2.set_density('g/cm3', 10.41)
        UO2.add_element('U', 1.0, enrichment= 100 * parms['enrichment'])
        UO2.add_nuclide('O16', 2.0)
        # Uranium Carbide
        UC = openmc.Material(name='UC')
        UC.set_density('g/cm3', 13.0)
        UC.add_element('U', 1.0,  enrichment= 100 * parms['enrichment'])
        UC.add_element('N', 1.0)
        
        #UCO: Mixed uranium dioxide (UO2) and uranium carbide (UC)
        uo2_atom_fraction = 0.7 # Mixing UO2 and UC by atom fraction
        fuel = openmc.Material.mix_materials([UO2, UC], [uo2_atom_fraction, 1- uo2_atom_fraction], 'ao',name='fuel') # mixing UO2 and UC by atom fraction
        
        fuel.temperature = operating_temperature
        fuel.depletable = True
        all_mats.append(fuel)
          
        #% --- Carbon buffer layer:
        
        buffer = openmc.Material(name='buffer') #mat buffer   -1.0400   rgb 0 200 200  moder grph_f 6012    tmp  operating_temperature
        buffer.add_nuclide('C12',    1.0, percent_type='ao')
        buffer.add_s_alpha_beta('c_Graphite')
        buffer.set_density('g/cm3', 1.0400)
        buffer.temperature = operating_temperature
        all_mats.append(buffer)
        
        #% --- Pyrolytic carbon layer:
        
        PyC1 = openmc.Material(name='PyC1') #mat PyC1     -1.8820    rgb 0 200 200  moder grph_f 6012    tmp  operating_temperature
        PyC1.add_nuclide('C12',    1.0, percent_type='ao')
        PyC1.add_s_alpha_beta('c_Graphite')
        PyC1.set_density('g/cm3', 1.8820)
        PyC1.temperature = operating_temperature
        all_mats.append(PyC1)
        
        #% --- Pyrolytic carbon layer:
        
        PyC2 = openmc.Material(name='PyC2') #mat PyC2     -1.8820  rgb 0 200 200    moder grph_f 6012     tmp  operating_temperature
        PyC2.add_nuclide('C12',    1.0, percent_type='ao')
        PyC2.add_s_alpha_beta('c_Graphite')
        PyC2.set_density('g/cm3', 1.8820)
        PyC2.temperature = operating_temperature
        all_mats.append(PyC2)
        
        #% --- Silicon carbide layer:
        
        SiC = openmc.Material(name='SiC') #mat SiC     -3.1710  rgb 0 100 0     tmp  operating_temperature
        SiC.add_nuclide('Si28',   0.4611, percent_type='ao')
        SiC.add_nuclide('Si29',   0.0234, percent_type='ao')
        SiC.add_nuclide('Si30',   0.0154, percent_type='ao')
        SiC.add_nuclide('C12',    0.5, percent_type='ao')
        SiC.set_density('g/cm3', 3.1710)
        SiC.temperature = operating_temperature
        all_mats.append(SiC)
        
        #% --- Graphite matrix:
        #therm grph_f grph.84t 
        matrix_pin = openmc.Material(name='matrix_pin') #mat matrix_pin   -1.8060  rgb 220 220 220    moder grph_f 6012    tmp  operating_temperature
        matrix_pin.add_nuclide('C12',  0.9999997, percent_type='ao')
        matrix_pin.add_nuclide('B10',  0.0000003, percent_type='ao')
        matrix_pin.add_s_alpha_beta('c_Graphite')
        matrix_pin.set_density('g/cm3', 1.8060)
        matrix_pin.temperature = operating_temperature
        all_mats.append(matrix_pin)
        
        #therm grph_m grph.84t
        matrix = openmc.Material(name='matrix') #mat matrix     -1.8060 rgb 200 200 200    moder grph_m 6012    tmp  operating_temperature
        matrix.add_nuclide('C12',  0.9999997, percent_type='ao')
        matrix.add_nuclide('B10',  0.0000003, percent_type='ao')
        matrix.add_s_alpha_beta('c_Graphite')
        matrix.set_density('g/cm3', 1.8060)
        matrix.temperature = operating_temperature
        all_mats.append(matrix)
        
        #therm bemet be-met.84t 
        beryllium = openmc.Material(name='beryllium') #mat beryllium  -1.848   rgb 0 255 0 moder bemet 4009  tmp  operating_temperature
        beryllium.add_nuclide('Be9',  1.00, percent_type='ao')
        beryllium.add_s_alpha_beta('c_Be')
        beryllium.set_density('g/cm3', 1.848)
        beryllium.temperature = operating_temperature
        all_mats.append(beryllium)
        
        beryllium_drum = openmc.Material(name='beryllium_drum') #mat beryllium_drum  -1.848   rgb 50 220 50 moder bemet 4009  tmp  operating_temperature
        beryllium_drum.add_nuclide('Be9',  1.00, percent_type='ao')
        beryllium_drum.add_s_alpha_beta('c_Be')
        beryllium_drum.set_density('g/cm3', 1.848)
        beryllium_drum.temperature = operating_temperature
        all_mats.append(beryllium_drum)
        
        #%
        B4C_drum = openmc.Material(name='B4C_drum') #mat B4C_drum  -2.510  rgb 250 0 0   tmp  operating_temperature
        B4C_drum.add_nuclide('B10', parms["B10_at_frac_B"]*.8, percent_type='ao')
        B4C_drum.add_nuclide('B11', (1-parms["B10_at_frac_B"])*.8, percent_type='ao')
        B4C_drum.add_nuclide('C12', 0.2, percent_type='ao')
        B4C_drum.set_density('g/cm3', 2.510)
        B4C_drum.temperature = operating_temperature
        all_mats.append(B4C_drum)
        #%
        B4C_central = openmc.Material(name='B4C_central') #mat B4C_central  -1.25  rgb 150 20 20   tmp  operating_temperature
        B4C_central.add_nuclide('B10', 0.76, percent_type='ao')
        B4C_central.add_nuclide('B11', 0.04, percent_type='ao')
        B4C_central.add_nuclide('C12', 0.2, percent_type='ao')
        B4C_central.set_density('g/cm3', 1.25)
        B4C_central.temperature = operating_temperature
        all_mats.append(B4C_central)
        #% -----------------------------------------------------------------------------
        
        # -- End materials --
        
        
        #% --- Compact --------------------------------------------------------------------------------------
        #surf  1 hexyprism 0.0 0.0 13.376      0 200 %fueled core
        # surf_1 isn't used in the OpenMC model
        #surf_1 = openmc.model.HexagonalPrism(edge_length=13.376, orientation='x', origin=(0,0), boundary_type='transmission', corner_radius=0.0)
        #surf 91 hexyprism 0.0 0.0 130.0       0 200 %fueled core
        # OpenMC does not have a z-truncated hex prism
        # create an infinite hex prism, also note OpenMC uses edge length while Serpent uses half width
        surf_91 = openmc.model.HexagonalPrism(edge_length=parms["reflector_width"]/2 * 2 / np.sqrt(3), orientation='x', origin=(0,0), boundary_type='transmission', corner_radius=0.0)
        # Create another slightly larger hexprism and truncating planes
        # What is universe zero in the Serpent model will be put inside a cell made of these surfaces
        surf_91o = openmc.model.HexagonalPrism(edge_length=(surf_91.plane_max.y0 + 0.1) * 2 / np.sqrt(3), orientation='x', origin=(0,0), boundary_type='vacuum', corner_radius=0.0)
        surf_91l = openmc.ZPlane(z0=0, boundary_type='vacuum')
        surf_91u = openmc.ZPlane(z0=2*parms["axial_reflector_height"] + parms["active_fuel_height"], boundary_type='vacuum')
        
        #surf 95l pz  20
        surf_95l = openmc.ZPlane(z0=parms["axial_reflector_height"])
        #surf 95u pz  180
        surf_95u = openmc.ZPlane(z0=parms["axial_reflector_height"] + parms["active_fuel_height"])
        #surf 96l pz  18.0
        surf_96l = openmc.ZPlane(z0=parms["axial_reflector_height"] - 2.)
        #surf 96u pz  182.0
        surf_96u = openmc.ZPlane(z0=parms["axial_reflector_height"] + parms["active_fuel_height"] + 2.)
        
        #surf  2 cyl 0.0 0.0 1.00   0.0 200
        surf_2 = openmc.ZCylinder(0, 0, parms["compact_radius"])
        #surf  3 cyl 0.0 0.0 0.825  0.0 200
        surf_3 = openmc.ZCylinder(0, 0, parms["moderator_radius"])
        #surf 35 cyl 0.0 0.0 0.875  0.0 200
        surf_35 = openmc.ZCylinder(0, 0, 0.875)
        #surf 36 cyl 0.0 0.0 0.900  0.0 200
        surf_36 = openmc.ZCylinder(0, 0, 0.900)
        #surf 3g cyl 0.0 0.0 0.920  0.0 200
        surf_3g = openmc.ZCylinder(0, 0, parms["moderator_radius"] + 0.095)
        #surf 5g cyl 0.0 0.0 1.07   0.0 200
        surf_5g = openmc.ZCylinder(0, 0, 1.07)
        #surf  5 cyl 0.0 0.0 1.05   0.0 200
        surf_5 = openmc.ZCylinder(0, 0, 1.05)
        #surf 51 cyl 0.0 0.0 0.97   0.0 200
        surf_51 = openmc.ZCylinder(0, 0, 0.970)
        #surf 52 cyl 0.0 0.0 0.90   0.0 200
        surf_52 = openmc.ZCylinder(0, 0, 0.90)
        #surf 53 cyl 0.0 0.0 0.80   0.0 200
        surf_53 = openmc.ZCylinder(0, 0, 0.80)
        
        #% infinite cells defining material universes
        # OpenMC cells are made infinite by not assigning a region
        #cell 51 802 moderator  -inf
        cell_51 = openmc.Cell(fill=moderator)
        uni_802 = openmc.Universe(cells=[cell_51])
        #cell 52 33  matrix_pin -inf
        cell_52 = openmc.Cell(fill=matrix_pin)
        uni_33 = openmc.Universe(cells=[cell_52])
        #cell 53 803    matrix  -inf
        cell_53 = openmc.Cell(fill=matrix)
        uni_803 = openmc.Universe(cells=[cell_53])
        #cell 55 804       air  -inf
        cell_55 = openmc.Cell(fill=air)
        uni_804 = openmc.Universe(cells=[cell_55])
        #cell 56 805  beryllium -inf
        cell_56 = openmc.Cell(fill=beryllium)
        uni_805 = openmc.Universe(cells=[cell_56])
        #cell 57 806  shell_mod -inf
        cell_57 = openmc.Cell(fill=shell_mod)
        uni_806 = openmc.Universe(cells=[cell_57])
        #cell 64 813  shell_hp  -inf
        cell_64 = openmc.Cell(fill=shell_hp)
        uni_813 = openmc.Universe(cells=[cell_64])
        #cell 58 820  shell_air_center -inf
        cell_58 = openmc.Cell(fill=shell_air_center)
        uni_820 = openmc.Universe(cells=[cell_58])
        
        #% mixed wick and coo_vap and liqu
        #cell 65 815 hp_vp_liq_wick -inf
        cell_65 = openmc.Cell(fill=hp_vp_liq_wick)
        uni_815 = openmc.Universe(cells=[cell_65])
        
        #% mixed air and ss in moderator
        #cell 68 816 shell_air_mod -inf
        cell_68 = openmc.Cell(fill=shell_air_mod)
        uni_816 = openmc.Universe(cells=[cell_68])
        
        #% mixed air and ss in heatpipe
        #cell 72 817 shell_air_hp   -inf
        cell_72 = openmc.Cell(fill=shell_air_hp)
        uni_817 = openmc.Universe(cells=[cell_72])
        
        #%  control drums 
        #cell 61 810 beryllium_drum -inf
        cell_61 = openmc.Cell(fill=beryllium_drum)
        uni_810 = openmc.Universe(cells=[cell_61])
        #cell 62 811   B4C_drum -inf
        cell_62 = openmc.Cell(fill=B4C_drum)
        uni_811 = openmc.Universe(cells=[cell_62])
        #cell 63 812   B4C_central -inf
        cell_63 = openmc.Cell(fill=B4C_central)
        uni_812 = openmc.Universe(cells=[cell_63])

        #beginning of pin cells
        #% Yan simplified model removed caps for the HP and moderator pins extended into the Be reflectors 
        #% moderator
        #cell 13   2 fill 802    -3      95l -95u % active length
        cell_13 = openmc.Cell(fill=uni_802, region=(-surf_3 & +surf_95l & -surf_95u)) # moderator material
        #cell 14   2 fill 816     3 -3g  95l -95u
        cell_14 = openmc.Cell(fill=uni_816, region=(+surf_3 & -surf_3g & +surf_95l & -surf_95u)) #shell_air_mod:mixed air and ss in moderator, moderator clad
        #cell 15   2 fill 803     3g     95l -95u
        cell_15 = openmc.Cell(fill=uni_803, region=(+surf_3g & +surf_95l & -surf_95u)) #graphite matrix
        #cell 151  2 fill 805       -36  96l -95l % 0.05 cm shell
        cell_151 = openmc.Cell(fill=uni_805, region=(-surf_36 & +surf_96l & -surf_95l))
        #cell 152  2 fill 805       -36  95u -96u % 0.05 cm shell
        cell_152 = openmc.Cell(fill=uni_805, region=(-surf_36 & +surf_95u & -surf_96u))
        #cell 151a 2 fill 805   -3g  36  96l -95l % 0.05 cm shell
        cell_151a = openmc.Cell(fill=uni_805, region=(-surf_3g & +surf_36 & +surf_96l & -surf_95l))
        #cell 152a 2 fill 805   -3g  36  95u -96u % 0.05 cm shell
        cell_152a = openmc.Cell(fill=uni_805, region=(-surf_3g & +surf_36 & +surf_95u & -surf_96u))
        #cell 153  2 fill 805        3g  96l -95l % 0.05 cm shell
        cell_153 = openmc.Cell(fill=uni_805, region=(+surf_3g & +surf_96l & -surf_95l))
        #cell 154  2 fill 805        3g  95u -96u % 0.05 cm shell
        cell_154 = openmc.Cell(fill=uni_805, region=(+surf_3g & +surf_95u & -surf_96u))
        #cell 155  2 fill 805                 96u
        cell_155 = openmc.Cell(fill=uni_805, region=(+surf_96u))
        #cell 156  2 fill 805                -96l
        cell_156 = openmc.Cell(fill=uni_805, region=(-surf_96l))
        uni_2 = openmc.Universe(cells=[cell_13,cell_14,cell_15,cell_151,cell_152,cell_151a,cell_152a,cell_153,cell_154,cell_155,cell_156])
        
        #% heat pipe cell
        #cell 16i  1 fill 815      -51 95l
        cell_16i = openmc.Cell(fill=uni_815, region=(-surf_51 & +surf_95l))
        #cell 16o  1 fill 817   51 -5g 95l
        cell_16o = openmc.Cell(fill=uni_817, region=(+surf_51 & -surf_5g & +surf_95l))
        #cell 17   1 fill 803       5g 95l -95u
        cell_17 = openmc.Cell(fill=uni_803, region=(+surf_5g & +surf_95l & -surf_95u))
        #cell 175  1 fill 805       5g 95u
        cell_175 = openmc.Cell(fill=uni_805, region=(+surf_5g & +surf_95u))
        #cell 176  1 fill 805         -96l
        cell_176 = openmc.Cell(fill=uni_805, region=(-surf_96l))
        #cell 177  1 fill 805      -5  96l -95l
        cell_177 = openmc.Cell(fill=uni_805, region=(-surf_5 & +surf_96l & -surf_95l))
        #cell 178  1 fill 805       5g 96l -95l
        cell_178 = openmc.Cell(fill=uni_805, region=(+surf_5g & +surf_96l & -surf_95l))
        #cell 178a 1 fill 805   -5g 5  96l -95l
        cell_178a = openmc.Cell(fill=uni_805, region=(-surf_5g & +surf_5 & +surf_96l & -surf_95l))
        uni_1 = openmc.Universe(cells=[cell_16i,cell_16o,cell_17,cell_175,cell_176,cell_177,cell_178,cell_178a])
        
        cell_18 = openmc.Cell(fill=uni_820)
        uni_8 = openmc.Universe(cells=[cell_18])
        
        #% monolith filled cell
        #cell 19  9 fill 803     -inf
        cell_19 = openmc.Cell(fill=uni_803)
        uni_9 = openmc.Universe(cells=[cell_19])
        
        #% monolith filled cell
        #cell 20 10 fill 805           -inf
        cell_20 = openmc.Cell(fill=uni_805)
        uni_10 = openmc.Universe(cells=[cell_20])
        
        #% monolith filled cell
        #% cell 21 11 fill 804           -inf
        #cell 21 11 fill 820           -inf
        cell_21 = openmc.Cell(fill=uni_820)
        uni_11 = openmc.Universe(cells=[cell_21])

        #lat 80  2  0.0 0.0 1 1 26.752
        # 10  % Be
        # Single element lattice? Use an "infinite" cell for now
        cell_lat_80 = openmc.Cell(fill=uni_10)
        uni_80 = openmc.Universe(cells=[cell_lat_80])
        
        #lat 90  2  0.0 0.0 1 1 26.752
        #  8  % air
        # Another single element lattice, using an "infinite" cell
        cell_lat_90 = openmc.Cell(fill=uni_8)
        uni_90 = openmc.Universe(cells=[cell_lat_90])
        
        #lat 91  2  0.0 0.0 1 1 26.752
        #  11  % air o  B4C
        # Another single element lattice, using an "infinite" cell
        cell_lat_91 = openmc.Cell(fill=uni_11)
        uni_91 = openmc.Universe(cells=[cell_lat_91])
        
        #% fuel
        #cell 11   3 fill 801        -2  95l -95u % fuel compact
        # cell_11 is the fuel compact containing TRISO particles. Syntax for this doesn't compare 1:1 with Serpent.
        # The Serpent model defines the fuel compact using surf_2 (a RCC that spans the full z of the problem space)
        # and further truncating it with surf_95l and surf_95u. The OpenMC TRISO particle packing and lattice algorithms
        # need a single RCC without further truncations.
        packing_divs_height = (surf_95u.z0 - surf_95l.z0)/axial_divs
        packing_div_surf = openmc.model.RightCircularCylinder((0.0,0.0,-packing_divs_height/2), packing_divs_height, surf_2.r, axis='z')
        packing_div_reg = -packing_div_surf
        if low_pf == True:
            print("RUNNING LOW PF FOR DEBUG")
            triso_centers = openmc.model.pack_spheres(radius=4.2750e-02, region=packing_div_reg, pf=0.01) # Calculate TRISO locations using OpenMC functionality
        else:
            triso_centers = openmc.model.pack_spheres(radius=4.2750e-02, region=packing_div_reg, pf=0.40)
        uni_900_spheres = [openmc.Sphere(r=r) for r in [2.1250e-02,3.1250e-02,3.5250e-02,3.8750e-02,4.2750e-02]]
        
        #complications arise from depletion divisions
        adiv_fuelmat_list = [[0 for _ in range(axial_divs)] for _ in range(3)]
        adiv_univ_list = [[0 for _ in range(axial_divs)] for _ in range(3)]
        flake_univ_list = [0 for _ in range(3)]
        print("Making depletable_regions")
        for nflake in range(3): #30 flakes in core
            for adiv in range(axial_divs):
                print(nflake, "/", 2, adiv, "/", axial_divs - 1)
                adiv_fuelmat_list[nflake][adiv] = fuel.clone()
                uni_900_cells = [openmc.Cell(fill=adiv_fuelmat_list[nflake][adiv], region=-uni_900_spheres[0]),
                                 openmc.Cell(fill=buffer, region=+uni_900_spheres[0] & -uni_900_spheres[1]),
                                 openmc.Cell(fill=PyC1, region=+uni_900_spheres[1] & -uni_900_spheres[2]),
                                 openmc.Cell(fill=SiC, region=+uni_900_spheres[2] & -uni_900_spheres[3]),
                                 openmc.Cell(fill=PyC2, region=+uni_900_spheres[3] & -uni_900_spheres[4])]
                uni_900 = openmc.Universe(cells=uni_900_cells)
                trisos = [openmc.model.TRISO(uni_900_spheres[-1].r, uni_900, center) for center in triso_centers]
                lower_left, upper_right = packing_div_reg.bounding_box
                shape = (2, 2, int(upper_right[2] - lower_left[2]))
                pitch = (upper_right - lower_left)/shape
                triso_lattice = openmc.model.create_triso_lattice(trisos, lower_left, pitch, shape, matrix_pin)
                cell_triso_lattice = openmc.Cell(fill=triso_lattice)
                adiv_univ_list[nflake][adiv] = openmc.Universe(cells=[cell_triso_lattice])

            stacked_packing_division_lattice = openmc.RectLattice()
            stacked_packing_division_lattice.pitch = (1e3, 1e3, packing_divs_height)
            stacked_packing_division_lattice.lower_left = (-1e3/2, -1e3/2, surf_95l.z0 - packing_divs_height)#packing_div_reg.bounding_box[0]
            universes = [[[uni_80]]] + [[[adiv_univ_list[nflake][adiv]]] for adiv in range(axial_divs)] + [[[uni_80]]]
            stacked_packing_division_lattice.universes = universes
            stacked_packing_division_cell = openmc.Cell(fill = stacked_packing_division_lattice)
            uni_801 = openmc.Universe(cells=[stacked_packing_division_cell])
            cell_12 = openmc.Cell(fill=uni_803, region=(+surf_2 & +surf_95l & -surf_95u))
            cell_125 = openmc.Cell(fill=uni_805, region=(+surf_95u))
            cell_126 = openmc.Cell(fill=uni_805, region=(-surf_95l))
            cell_11 = openmc.Cell(fill=uni_801, region=-surf_2 & +surf_95l & -surf_95u)
            uni_3 = openmc.Universe(cells=[cell_11, cell_12, cell_125, cell_126])
        
            #% --- Assembly lattice
            
            #lat 20  2  0.0 0.0 19 19 2.3
            #% New Assembly Configuration - 2 time less moderator - more fuel
            # 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9
            #  9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9
            #   9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9
            #    9 9 9 9 9 9 9 9 9 1 3 1 3 1 3 1 9 9 9
            #     9 9 9 9 9 9 9 9 3 3 2 3 2 3 2 3 9 9 9
            #      9 9 9 9 9 9 9 1 2 1 3 1 3 1 3 1 9 9 9 
            #       9 9 9 9 9 9 3 3 3 3 2 3 2 3 2 3 9 9 9
            #        9 9 9 9 9 1 2 1 2 1 3 1 3 1 3 1 9 9 9 
            #         9 9 9 9 3 3 3 3 3 3 2 3 2 3 2 3 9 9 9
            #          9 9 9 1 2 1 2 1 2 1 3 1 3 1 3 1 9 9 9
            #           9 9 9 3 3 3 3 3 3 2 3 2 3 2 3 9 9 9 9
            #            9 9 9 1 2 1 2 1 3 1 3 1 3 1 9 9 9 9 9 
            #             9 9 9 3 3 3 3 2 3 2 3 2 3 9 9 9 9 9 9
            #              9 9 9 1 2 1 3 1 3 1 3 1 9 9 9 9 9 9 9 
            #               9 9 9 3 3 2 3 2 3 2 3 9 9 9 9 9 9 9 9
            #                9 9 9 1 3 1 3 1 3 1 9 9 9 9 9 9 9 9 9
            #                 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 
            #                  9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9
            #                   9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9
            # OpenMC lattices are defined via rings.
            lat_20 = openmc.HexLattice()
            lat_20.pitch = [parms["pin_pitch"]]
            lat_20.outer = uni_9
            lat_20.center = (0.0, 0.0)
            lat_20.orientation = 'x'
            # ORIGINAL
            lat_20.universes = [[x for x in itertools.chain.from_iterable(itertools.repeat([uni_1, uni_3], 18))], # 18 * 2 = 36
                                [x for x in itertools.chain.from_iterable(itertools.repeat([uni_3, uni_2], 15))], # 15 * 2 = 30
                                [x for x in itertools.chain.from_iterable(itertools.repeat([uni_1, uni_3], 12))], # 12 * 2 = 24
                                [x for x in itertools.chain.from_iterable(itertools.repeat([uni_3, uni_2], 9))],  # 9 * 2 = 18
                                [x for x in itertools.chain.from_iterable(itertools.repeat([uni_1, uni_3], 6))],  # 6 * 2 = 12
                                [x for x in itertools.chain.from_iterable(itertools.repeat([uni_3, uni_2], 3))],  # 3 * 2 = 6
                                [uni_1]]
            cell_lat_20 = openmc.Cell(fill=lat_20)
            #trans 20 0.0 0.0 0.0 0.0 0.0 30.
            cell_lat_20.rotation= (0.0,0.0,-30.0)
            flake_univ_list[nflake] = openmc.Universe(cells=[cell_lat_20])
          
        

        lat_50 = openmc.HexLattice()
        lat_50.pitch = [parms["flake_width"]]
        lat_50.outer = uni_90
        lat_50.center = (0.0, 0.0)
        lat_50.orientation = 'x'
        #assembly-wide depletable regions sit in a ring
        lat_50.universes = [[x for x in itertools.chain.from_iterable(itertools.repeat([uni_90, uni_80], 12))],        # 12 * 2 = 24
                            [x for x in itertools.chain.from_iterable(itertools.repeat([uni_90, flake_univ_list[0], flake_univ_list[0]], 6))], # 6 * 3 = 18
                            [x for x in itertools.chain.from_iterable(itertools.repeat([flake_univ_list[1]], 12))],                # 12 * 1 = 12
                            [x for x in itertools.chain.from_iterable(itertools.repeat([flake_univ_list[2]], 6))],                 # 6 * 1 = 6
                            [uni_91]]
        cell_lat_50 = openmc.Cell(fill=lat_50)
        uni_50 = openmc.Universe(cells=[cell_lat_50])
        
        
        #% -----------DRUMS DEFINITTION---------------------------------------------------------------------------------------
        drum_outer_r = (parms["flake_width"] - 0.252)/2
        drum_inner_r = drum_outer_r - 1.
        
        #surf 731 cyl 80.2560   0.00 13.2500   0.00 200.00
        surf_731 = openmc.model.RightCircularCylinder((parms["flake_width"]*3,0.0,0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_outer_r, axis='z')
        #surf 831 cyl 80.2560   0.00 12.2500   0.00 200.00
        surf_831 = openmc.model.RightCircularCylinder((parms["flake_width"]*3,0.0,0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_inner_r, axis='z')
        if most_effective_rods_in:
            surf_771 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 180 + 0, np.array([parms["flake_width"]*3, 0])))
        else:
            surf_771 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 180 + Tdict["DrumRot"], np.array([parms["flake_width"]*3, 0])))
        #cell 801  731 fill 810   -731   -771
        cell_801 = openmc.Cell(fill=uni_810, region=(-surf_731 & -surf_771))
        #cell 802  731 fill 811    831  -731  771
        cell_802 = openmc.Cell(fill=uni_811, region=(+surf_831 & -surf_731 & +surf_771))
        #cell 803  731 fill 810   -831   771
        cell_803 = openmc.Cell(fill=uni_810, region=(-surf_831 & +surf_771))
        uni_731 = openmc.Universe(cells=[cell_801, cell_802, cell_803])
        #surf 732 cyl 40.1280  69.50 13.2500   0.00 200.00
        surf_732 = openmc.model.RightCircularCylinder((parms["flake_width"]*3/2,1.5*np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_outer_r, axis='z')
        #surf 832 cyl 40.1280  69.50 12.2500   0.00 200.00
        surf_832 = openmc.model.RightCircularCylinder((parms["flake_width"]*3/2,1.5*np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_inner_r, axis='z')
        surf_772 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 240 + Tdict["DrumRot"], np.array([parms["flake_width"]*3/2, 1.5*np.sqrt(3)*parms["flake_width"]])))
        #cell 806  732 fill 810   -732   -772
        cell_806 = openmc.Cell(fill=uni_810, region=(-surf_732 & -surf_772))
        #cell 807  732 fill 811    832  -732  772
        cell_807 = openmc.Cell(fill=uni_811, region=(+surf_832 & -surf_732 & +surf_772))
        #cell 808  732 fill 810   -832   772
        cell_808 = openmc.Cell(fill=uni_810, region=(-surf_832 & +surf_772))
        uni_732 = openmc.Universe(cells=[cell_806, cell_807, cell_808])
        #surf 733 cyl -40.1280  69.50 13.2500   0.00 200.00
        surf_733 = openmc.model.RightCircularCylinder((-parms["flake_width"]*3/2,1.5*np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_outer_r, axis='z')
        #surf 833 cyl -40.1280  69.50 12.2500   0.00 200.00
        surf_833 = openmc.model.RightCircularCylinder((-parms["flake_width"]*3/2,1.5*np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_inner_r, axis='z')
        #surf 773 plane 1.0000  -1.73 0.0 -177.84
        surf_773 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 300 + Tdict["DrumRot"], np.array([-parms["flake_width"]*3/2, 1.5*np.sqrt(3)*parms["flake_width"]])))
        #cell 811  733 fill 810   -733   773
        cell_811 = openmc.Cell(fill=uni_810, region=(-surf_733 & +surf_773))
        #cell 812  733 fill 811    833  -733  -773
        cell_812 = openmc.Cell(fill=uni_811, region=(+surf_833 & -surf_733 & -surf_773))
        #cell 813  733 fill 810   -833   -773
        cell_813 = openmc.Cell(fill=uni_810, region=(-surf_833 & -surf_773))
        uni_733 = openmc.Universe(cells=[cell_811, cell_812, cell_813])
        #surf 734 cyl -80.2560   0.00 13.2500   0.00 200.00
        surf_734 = openmc.model.RightCircularCylinder((-parms["flake_width"]*3,0.0,0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_outer_r, axis='z')
        #surf 834 cyl -80.2560   0.00 12.2500   0.00 200.00
        surf_834 = openmc.model.RightCircularCylinder((-parms["flake_width"]*3,0.0,0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_inner_r, axis='z')
        #surf 774 plane 1.0000  -0.00 0.0 -88.92
        surf_774 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 0 + Tdict["DrumRot"], np.array([-parms["flake_width"]*3, 0.0])))
        #cell 816  734 fill 810   -734   774
        cell_816 = openmc.Cell(fill=uni_810, region=(-surf_734 & +surf_774))
        #cell 817  734 fill 811    834  -734  -774
        cell_817 = openmc.Cell(fill=uni_811, region=(+surf_834 & -surf_734 & -surf_774))
        #cell 818  734 fill 810   -834   -774
        cell_818 = openmc.Cell(fill=uni_810, region=(-surf_834 & -surf_774))
        uni_734 = openmc.Universe(cells=[cell_816, cell_817, cell_818])
        #surf 735 cyl -40.1280 -69.50 13.2500   0.00 200.00
        surf_735 = openmc.model.RightCircularCylinder((-parms["flake_width"]*3/2,-1.5*np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_outer_r, axis='z')
        #surf 835 cyl -40.1280 -69.50 12.2500   0.00 200.00
        surf_835 = openmc.model.RightCircularCylinder((-parms["flake_width"]*3/2,-1.5*np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_inner_r, axis='z')
        #surf 775 plane 1.0000   1.73 0.0 -177.84
        surf_775 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 60 + Tdict["DrumRot"], np.array([-parms["flake_width"]*3/2, -1.5*np.sqrt(3)*parms["flake_width"]])))
        #cell 821  735 fill 810   -735   775
        cell_821 = openmc.Cell(fill=uni_810, region=(-surf_735 & +surf_775))
        #cell 822  735 fill 811    835  -735  -775
        cell_822 = openmc.Cell(fill=uni_811, region=(+surf_835 & -surf_735 & -surf_775))
        #cell 823  735 fill 810   -835   -775
        cell_823 = openmc.Cell(fill=uni_810, region=(-surf_835 & -surf_775))
        uni_735 = openmc.Universe(cells=[cell_821, cell_822, cell_823])
        #surf 736 cyl 40.1280 -69.50 13.2500   0.00 200.00
        surf_736 = openmc.model.RightCircularCylinder((parms["flake_width"]*3/2,-1.5*np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_outer_r, axis='z')
        #surf 836 cyl 40.1280 -69.50 12.2500   0.00 200.00
        surf_836 = openmc.model.RightCircularCylinder((parms["flake_width"]*3/2,-1.5*np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_inner_r, axis='z')
        #surf 776 plane 1.0000  -1.73 0.0 177.84
        surf_776 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 120 + Tdict["DrumRot"], np.array([parms["flake_width"]*3/2, -1.5*np.sqrt(3)*parms["flake_width"]])))
        #cell 826  736 fill 810   -736   -776
        cell_826 = openmc.Cell(fill=uni_810, region=(-surf_736 & -surf_776))
        #cell 827  736 fill 811    836  -736  776
        cell_827 = openmc.Cell(fill=uni_811, region=(+surf_836 & -surf_736 & +surf_776))
        #cell 828  736 fill 810   -836   776
        cell_828 = openmc.Cell(fill=uni_810, region=(-surf_836 & +surf_776))
        uni_736 = openmc.Universe(cells=[cell_826, cell_827, cell_828])
        #surf 737 cyl 80.2563  46.34 13.2500   0.00 200.00
        surf_737 = openmc.model.RightCircularCylinder((parms["flake_width"]*3,np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_outer_r, axis='z')
        #surf 837 cyl 80.2563  46.34 12.2500   0.00 200.00
        surf_837 = openmc.model.RightCircularCylinder((parms["flake_width"]*3,np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_inner_r, axis='z')
        #surf 777 plane 1.0000   0.58 0.0 117.01
        surf_777 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 210 + Tdict["DrumRot"], np.array([parms["flake_width"]*3, np.sqrt(3)*parms["flake_width"]])))
        #cell 831  737 fill 810   -737   -777
        cell_831 = openmc.Cell(fill=uni_810, region=(-surf_737 & -surf_777))
        #cell 832  737 fill 811    837  -737  777
        cell_832 = openmc.Cell(fill=uni_811, region=(+surf_837 & -surf_737 & +surf_777))
        #cell 833  737 fill 810   -837   777
        cell_833 = openmc.Cell(fill=uni_810, region=(-surf_837 & +surf_777))
        uni_737 = openmc.Universe(cells=[cell_831, cell_832, cell_833])
        #surf 738 cyl 0.0000  92.67 13.2500   0.00 200.00
        surf_738 = openmc.model.RightCircularCylinder((0.0000,2*np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_outer_r, axis='z')
        #surf 838 cyl 0.0000  92.67 12.2500   0.00 200.00
        surf_838 = openmc.model.RightCircularCylinder((0.0000,2*np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_inner_r, axis='z')
        #surf 778 plane 0.0000   1.00 0.0 101.33
        surf_778 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 270 + Tdict["DrumRot"], np.array([0.0, 2*np.sqrt(3)*parms["flake_width"]])))
        #cell 836  738 fill 810   -738   -778
        cell_836 = openmc.Cell(fill=uni_810, region=(-surf_738 & -surf_778))
        #cell 837  738 fill 811    838  -738  778
        cell_837 = openmc.Cell(fill=uni_811, region=(+surf_838 & -surf_738 & +surf_778))
        #cell 838  738 fill 810   -838   778
        cell_838 = openmc.Cell(fill=uni_810, region=(-surf_838 & +surf_778))
        uni_738 = openmc.Universe(cells=[cell_836, cell_837, cell_838])
        #surf 739 cyl -80.2563  46.34 13.2500   0.00 200.00
        surf_739 = openmc.model.RightCircularCylinder((-parms["flake_width"]*3,np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_outer_r, axis='z')
        #surf 839 cyl -80.2563  46.34 12.2500   0.00 200.00
        surf_839 = openmc.model.RightCircularCylinder((-parms["flake_width"]*3,np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_inner_r, axis='z')
        #surf 779 plane 1.0000  -0.58 0.0 -117.01
        surf_779 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 330 + Tdict["DrumRot"], np.array([-parms["flake_width"]*3, np.sqrt(3)*parms["flake_width"]])))
        #cell 841  739 fill 810   -739   779
        cell_841 = openmc.Cell(fill=uni_810, region=(-surf_739 & +surf_779))
        #cell 842  739 fill 811    839  -739  -779
        cell_842 = openmc.Cell(fill=uni_811, region=(+surf_839 & -surf_739 & -surf_779))
        #cell 843  739 fill 810   -839   -779
        cell_843 = openmc.Cell(fill=uni_810, region=(-surf_839 & -surf_779))
        uni_739 = openmc.Universe(cells=[cell_841, cell_842, cell_843])
        #surf 740 cyl -80.2563 -46.34 13.2500   0.00 200.00
        surf_740 = openmc.model.RightCircularCylinder((-parms["flake_width"]*3,-np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_outer_r, axis='z')
        #surf 840 cyl -80.2563 -46.34 12.2500   0.00 200.00
        surf_840 = openmc.model.RightCircularCylinder((-parms["flake_width"]*3,-np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_inner_r, axis='z')
        #surf 780 plane 1.0000   0.58 0.0 -117.01
        surf_780 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 30 + Tdict["DrumRot"], np.array([-parms["flake_width"]*3, -np.sqrt(3)*parms["flake_width"]])))
        #cell 846  740 fill 810   -740   780
        cell_846 = openmc.Cell(fill=uni_810, region=(-surf_740 & +surf_780))
        #cell 847  740 fill 811    840  -740  -780
        cell_847 = openmc.Cell(fill=uni_811, region=(+surf_840 & -surf_740 & -surf_780))
        #cell 848  740 fill 810   -840   -780
        cell_848 = openmc.Cell(fill=uni_810, region=(-surf_840 & -surf_780))
        uni_740 = openmc.Universe(cells=[cell_846, cell_847, cell_848])
        #surf 741 cyl -0.0000 -92.67 13.2500   0.00 200.00
        surf_741 = openmc.model.RightCircularCylinder((0.0,-2*np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_outer_r, axis='z')
        #surf 841 cyl -0.0000 -92.67 12.2500   0.00 200.00
        surf_841 = openmc.model.RightCircularCylinder((0.0,-2*np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_inner_r, axis='z')
        #surf 781 plane 0.0000   1.00 0.0 -101.33
        surf_781 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 90 + Tdict["DrumRot"], np.array([0, -2*np.sqrt(3)*parms["flake_width"]])))
        #cell 851  741 fill 810   -741   781
        cell_851 = openmc.Cell(fill=uni_810, region=(-surf_741 & +surf_781))
        #cell 852  741 fill 811    841  -741  -781
        cell_852 = openmc.Cell(fill=uni_811, region=(+surf_841 & -surf_741 & -surf_781))
        #cell 853  741 fill 810   -841   -781
        cell_853 = openmc.Cell(fill=uni_810, region=(-surf_841 & -surf_781))
        uni_741 = openmc.Universe(cells=[cell_851, cell_852, cell_853])
        #surf 742 cyl 80.2563 -46.34 13.2500   0.00 200.00
        surf_742 = openmc.model.RightCircularCylinder((parms["flake_width"]*3,-np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_outer_r, axis='z')
        #surf 842 cyl 80.2563 -46.34 12.2500   0.00 200.00
        surf_842 = openmc.model.RightCircularCylinder((parms["flake_width"]*3,-np.sqrt(3)*parms["flake_width"],0.0), parms["active_fuel_height"] + 2*parms["axial_reflector_height"], drum_inner_r, axis='z')
        #surf 782 plane 1.0000  -0.58 0.0 117.01
        surf_782 = openmc.Plane(*drum_parameters_to_eq(12.25, parms["coating_angle"], 150 + Tdict["DrumRot"], np.array([parms["flake_width"]*3, -np.sqrt(3)*parms["flake_width"]])))
        #cell 856  742 fill 810   -742   -782
        cell_856 = openmc.Cell(fill=uni_810, region=(-surf_742 & -surf_782))
        #cell 857  742 fill 811    842  -742  782
        cell_857 = openmc.Cell(fill=uni_811, region=(+surf_842 & -surf_742 & +surf_782))
        #cell 858  742 fill 810   -842   782
        cell_858 = openmc.Cell(fill=uni_810, region=(-surf_842 & +surf_782))
        uni_742 = openmc.Universe(cells=[cell_856, cell_857, cell_858])
        
        #% -----------FILL LATTICE AND DRUMS TO UNIVERSE 0---------------------------------------------------
        
        #cell 102 0 fill 50 -91 731 732 733 734 735 736 737 738 739 740 741 742
        cell_102 = openmc.Cell(fill=uni_50, region=(-surf_91 & +surf_731 & +surf_732 & +surf_733 &
                                                    +surf_734 & +surf_735 & +surf_736 & +surf_737 &
                                                    +surf_738 & +surf_739 & +surf_740 & +surf_741 &
                                                    +surf_742))
        #cell 180 0 fill       731 -731
        cell_180 = openmc.Cell(fill=uni_731, region=(-surf_731))
        #cell 181 0 fill       732 -732
        cell_181 = openmc.Cell(fill=uni_732, region=(-surf_732))
        #cell 182 0 fill       733 -733
        cell_182 = openmc.Cell(fill=uni_733, region=(-surf_733))
        #cell 183 0 fill       734 -734
        cell_183 = openmc.Cell(fill=uni_734, region=(-surf_734))
        #cell 184 0 fill       735 -735
        cell_184 = openmc.Cell(fill=uni_735, region=(-surf_735))
        #cell 185 0 fill       736 -736
        cell_185 = openmc.Cell(fill=uni_736, region=(-surf_736))
        #cell 186 0 fill       737 -737
        cell_186 = openmc.Cell(fill=uni_737, region=(-surf_737))
        #cell 187 0 fill       738 -738
        cell_187 = openmc.Cell(fill=uni_738, region=(-surf_738))
        #cell 188 0 fill       739 -739
        cell_188 = openmc.Cell(fill=uni_739, region=(-surf_739))
        #cell 189 0 fill       740 -740
        cell_189 = openmc.Cell(fill=uni_740, region=(-surf_740))
        #cell 190 0 fill       741 -741
        cell_190 = openmc.Cell(fill=uni_741, region=(-surf_741))
        #cell 191 0 fill       742 -742
        cell_191 = openmc.Cell(fill=uni_742, region=(-surf_742))
        #cell 104 0 outside  91
        # The hexprisms in OpenMC are infinite in z unlike Serpent
        # where they have z limits. I kept cell_102 infinite in order
        # to keep the above section as close to a 1:1 Serpent conversion
        # as possible.
        # Define cell_104 as void outside surf_91
        cell_104 = openmc.Cell(fill=None, region=(+surf_91))
        # Assembly uni_0
        uni_0 = openmc.Universe(cells=[cell_102,cell_180,cell_181,cell_182,
                                       cell_183,cell_184,cell_185,cell_186,
                                       cell_187,cell_188,cell_189,cell_190,
                                       cell_191,cell_104])
        # Stick uni_0 inside a hex prism with truncating z planes
        cell_00 = openmc.Cell(fill=uni_0, region=(-surf_91o & -surf_91u & +surf_91l))
        uni_00 = openmc.Universe(cells=[cell_00])
        #% --------------------------------------------------------------------------------------------------
        
        #####################################
        ##. Export geometry and materials
        #####################################
        geom = openmc.Geometry(uni_00)

        #######################
        ##. Settings to Run
        ########################
        
        settings = openmc.Settings()
        # Particles and batches set to match reference Serpent input so runtime can be compared.
        # Number of particles needed to generate cross sections with adequate statistics is TBD.
        settings.particles = particles
        settings.batches = batches 
        settings.inactive = inactive
        settings.output      = {'tallies': False, # turns off text tally output, hdf5 still produced
                                'summary': False} # the summary.h5 file is taking hours to write due to complexity of model with TRISO
        settings.temperature = {'default':700.0,
                                'method':'interpolation',
                                'range':(300.0, 1200.0)}
        # Source definition for first batch
        # Default is a isotropic Watt fission spectrum at (0,0,0). Move it to a point at core midplane
        initial_ksource = openmc.IndependentSource()
        initial_ksource.space = openmc.stats.Point(xyz=(0.0,0.0,parms["axial_reflector_height"] + parms["active_fuel_height"]/2))
        settings.source = initial_ksource
        
        
        #######################
        ##. Tallies
        ########################
        tallies = openmc.Tallies()
        if tally:
            pin_power_tally = openmc.Tally()
            pin_power_tally.scores = ["fission"]
            low_pinz = parms["axial_reflector_height"]
            upper_pinz = parms["axial_reflector_height"] + parms["active_fuel_height"]
            zmesh = openmc.RegularMesh()
            zmesh.dimension = (1400, 1400, 20)
            low_x = 3*parms["flake_width"]
            low_y = 2*parms["flake_width"]/np.sqrt(3)/2*5.5
            zmesh.lower_left = (-low_x, -low_y, low_pinz)
            zmesh.upper_right = (low_x, low_y, upper_pinz)
            pin_axial_position_filter = openmc.MeshFilter(zmesh)
            pin_power_tally.filters = [pin_axial_position_filter]
            tallies.append(pin_power_tally)
        
        material_colors = {
            air: 'white',  # rgb(255, 255, 255)
            shell_mod: 'grey',  # rgb(133, 133, 133)
            shell_hp: 'darkgrey',  # rgb(169, 169, 169)
            shell_air_mod: 'silver',  # rgb(192, 192, 192)
            shell_air_hp: 'lightgrey',  # rgb(211, 211, 211)
            shell_air_center: 'gainsboro',  # rgb(220, 220, 220)
            coo_vap: 'purple',  # rgb(128, 0, 128)
            coo_liq: 'green',  # rgb(0, 128, 0)
            wick: 'yellow',  # rgb(255, 255, 0)
            hp_vp_liq_wick: 'khaki',  # rgb(240, 230, 140)
            moderator: 'blueviolet',  # rgb(138, 43, 226)
            fuel: 'red',  # rgb(255, 0, 0)
            buffer: 'cyan',  # rgb(0, 255, 255)
            PyC1: 'teal',  # rgb(0, 128, 128)
            PyC2: 'darkcyan',  # rgb(0, 139, 139)
            SiC: 'darkgreen',  # rgb(0, 100, 0)
            matrix_pin: 'lightgray',  # rgb(211, 211, 211)
            matrix: 'darkgray',  # rgb(169, 169, 169)
            beryllium: 'magenta',  # rgb(0, 255, 0)
            beryllium_drum: 'limegreen',  # rgb(50, 205, 50)
            B4C_drum: 'crimson',  # rgb(220, 20, 60)
            B4C_central: 'darkred',  # rgb(139, 0, 0)
        }
        allfuel_mats = list(itertools.chain.from_iterable(adiv_fuelmat_list))
        for m in allfuel_mats:
            material_colors[m] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for a in allfuel_mats:
            all_mats.append(a)

        fuel_vol_per_flake = 4/3*np.pi*(2.1250e-02)**3*len(trisos)*63 #fuel vol in axial depletion region in cm^3
        for adiv in range(axial_divs):
            adiv_fuelmat_list[0][adiv].volume = fuel_vol_per_flake*12
            adiv_fuelmat_list[1][adiv].volume = fuel_vol_per_flake*12
            adiv_fuelmat_list[2][adiv].volume = fuel_vol_per_flake*6
        fuel.depletable = False
        print("Total Fuel Volume", fuel_vol_per_flake*(12+12+6)*axial_divs)


        # View of entire reactor at z=21 to match Serpent plot
        plot1 = openmc.Plot()
        plot1.colors = material_colors
        plot1.basis = 'xy'
        plot1.origin = (0.0,0.0,100+packing_divs_height/5)
        #plot1.origin = (0.0,0.0,199.0)
        plot1.width = (300.0, 300.0)
        #plot1.pixels = (1000, 1000)   # low fidelity for shorter runtime
        plot1.pixels = (2000, 2000) # high fidelity to match Serpent
        plot1.color_by = 'material'
        
        # View of entire reactor at y=5 to match Serpent plot
        plot2 = openmc.Plot()
        plot2.basis = 'xz'
        plot2.origin = (0.0,5.0,100.0)
        plot2.width = (280.0, 280.0)
        #plot2.pixels = (1000, 1000)   # low fidelity for shorter runtime
        plot2.pixels = (2000, 2000) # high fidelity to match Serpent
        plot2.color_by = 'material'
        plot2.colors = material_colors
        
        # Views of fuel assembly
        plot3 = openmc.Plot()
        plot3.basis = 'xy'
        plot3.origin = (parms["flake_width"],0.0,100+packing_divs_height/5)
        plot3.width = (parms["flake_width"], parms["flake_width"])
        plot3.pixels = (500, 500)
        plot3.color_by = 'material'
        plot3.colors = material_colors
        
        # Views of fuel compacts
        plot4 = openmc.Plot()
        plot4.basis = 'xy'
        plot4.origin = (parms["flake_width"],0.0,100+packing_divs_height/5)
        plot4.width = (parms["pin_pitch"]*3, parms["pin_pitch"]*3)
        plot4.pixels = (750, 750)
        plot4.color_by = 'material'
        plot4.colors = material_colors
        
        plots = openmc.Plots([plot1, plot2, plot3, plot4])

        #volume calculation
        #low_x = 3*parms["flake_width"]
        #low_y = 2*parms["flake_width"]/np.sqrt(3)/2*5.5
        #ll = (-low_x, -low_y, low_pinz)
        #ur = (low_x, low_y, upper_pinz)
        #vol_calc = openmc.VolumeCalculation(allfuel_mats, int(1e8), ll, ur)
        #settings.volume_calculations = [vol_calc]

        model = openmc.model.Model(geometry = geom, materials = all_mats, settings = settings, plots = plots, tallies = tallies)
        model.export_to_xml(directory = rundir)
        #calculate volumes
        #openmc.calculate_volumes(cwd = rundir, threads = 48)
        #print("VOLUME RUNTIME", time.time() - start)
        #print("ANA RESULTS")
        #for adiv in range(axial_divs):
        #    print(adiv_fuelmat_list[0][adiv].id, fuel_vol_per_flake*12)
        #    print(adiv_fuelmat_list[1][adiv].id, fuel_vol_per_flake*12)
        #    print(adiv_fuelmat_list[2][adiv].id, fuel_vol_per_flake*6)
        return model, allfuel_mats

    def postproc(self):
        #load the parameter dict
        with open(self.working_dir / Path("design_parameters.txt"), "r") as f:
            c = f.readlines()[1:]
        parms = {}
        for l in c:
            parms[l.split(",")[0]] = float(l.split(",")[1].strip())

        #shutdown margin
        sps = [f for f in Path(self.working_dir / Path("run_0")).iterdir() if "statepoint" in f.name]
        if len(sps) != 1:
            raise Exception("Multiple or no statepoint files found for run_0")
        deg0_k = openmc.StatePoint(sps[0]).keff.nominal_value
        delta_k2 = (1 - deg0_k)*1e5 # HFP to All Rods in (ARI) > 0

        sps = [f for f in Path(self.working_dir / Path("run_hzp_90")).iterdir() if "statepoint" in f.name]
        if len(sps) != 1:
            raise Exception("Multiple or no statepoint files found for run_hzp_90")
        deghzp90_k = openmc.StatePoint(sps[0]).keff.nominal_value
        delta_k1 = (1 - deghzp90_k)*1e5 # HFP to HZP < 0

        sps = [f for f in Path(self.working_dir / Path("run_meri_90")).iterdir() if "statepoint" in f.name]
        if len(sps) != 1:
            raise Exception("Multiple or no statepoint files found for run_hzp_90")
        degmeri90_k = openmc.StatePoint(sps[0]).keff.nominal_value
        delta_k3 = (1 - degmeri90_k)*1e5 # HFP to most effective rods in > 0
        shutdown_margin = delta_k1 + 0.9*(delta_k2 - delta_k3)
        
        #lifetime estimate
        sps = [f for f in Path(self.working_dir / Path("run_180")).iterdir() if "depletion_results" in f.name]
        if len(sps) != 1:
            raise Exception("Multiple or no depletion file found for run_180")
        time, k = openmc.deplete.Results(sps[0]).get_keff()
        time /= 31536000 #convert seconds to years
        k = k[:,0]
        lifetime = find_crossing_time(time, k)

        #BOL peaking factor
        sps = [f for f in Path(self.working_dir / Path("run_90")).iterdir() if "statepoint" in f.name]
        if len(sps) != 1:
            raise Exception("Multiple or no statepoint files found for run_0")
        tally = openmc.StatePoint(sps[0]).get_tally().get_reshaped_data(expand_dims = True).sum(4).sum(3)
        lz = parms["axial_reflector_height"]
        uz = parms["axial_reflector_height"] + parms["active_fuel_height"]
        xe = 3*parms["flake_width"]
        ye = 2*parms["flake_width"]/np.sqrt(3)/2*5.5
        ll = (-xe, -ye, lz)
        ur = (xe, ye, uz)
        pin_powers, z_coords = proc_pinpower(tally, ll, ur, parms["pin_pitch"], parms["flake_width"])
        write_pinpowers(self.working_dir, pin_powers, z_coords)
        ara_pin_powers = np.array(list(pin_powers.values()))

        Fq = ara_pin_powers.max()/ara_pin_powers.mean()
        fullpin = ara_pin_powers.sum(1)
        Fdh = fullpin.max()/fullpin.mean()

        # get peak heat flux
        q_doubleprime_avg = calculate_heat_flux(Power=2,N_flakes=30,N_compact=63,compact_radius=parms['compact_radius'],active_height=parms['active_fuel_height']) # MW / m^2
        q_doubleprim_max = Fq*q_doubleprime_avg
        # get peak power transfer
        q_avg = calculate_average_power_transfer(Power=2,N_flakes=30,N_compact=63) # MW
        q_max = Fdh* q_avg
        # get linear heat generation rate along the hottest rod
        q_prime_avg = calculate_average_q_prime(Power=2,N_flakes=30,N_compact=63,active_fuel_height = parms['active_fuel_height']) * 10**5 # kW / m
        q_prime_max_distr = ara_pin_powers[fullpin.argmax()]/ara_pin_powers.mean() * q_prime_avg
        objectives = {"Fq":Fq, "Fdh":Fdh, "shutdown_margin":shutdown_margin,'delta_k1':delta_k1,'delta_k2':delta_k2,'delta_k3':delta_k3,"lifetime":lifetime,'peak_heat_flux':q_doubleprim_max,'avg_heat_flux':q_doubleprime_avg,'peak_transfered_power':q_max,'avg_transfered_power':q_avg}
        write_dict(objectives, self.working_dir / Path("objectives.csv"))
        data = pd.DataFrame()
        data['q_prime_max_distribution'] = q_prime_max_distr
        data.index=z_coords
        data.to_csv(self.working_dir / Path("objectives_qprime.csv"),index_label='Z_coords')
        #print('--- columns:',pin_powers.keys())
        print('Loc of Fdh',fullpin.argmax(),list(pin_powers.keys())[fullpin.argmax()])
        

        data_k = pd.DataFrame()
        data_k['k'] = k
        data_k.index=time
        data_k.to_csv(self.working_dir / Path("objectives_keff.csv"),index_label='time')
    