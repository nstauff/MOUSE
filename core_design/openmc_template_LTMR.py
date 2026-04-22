# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
# Importing libraries
import openmc
import numpy as np
from core_design.openmc_materials_database import collect_materials_data
from core_design.utils import create_universe_plot, circle_area, create_cells
import copy


# **************************************************************************************************************************
#                                                Sec. 0 : Helper Functions
# **************************************************************************************************************************

"""
Helper functions are smaller, reusable functions ar defined to perform specific tasks,
and then used later to simplify and organize the code
"""

def create_pin_regions(params, pin_type):
    """
    Creating the pin regions
    @ In, params, dict, The parameters that are used to "fill in" input files with placeholders.
    @ In, pin_type, str, The type of pin ('moderator' or 'fuel').
    @ out, regions, dict, regions of the specified pin
    """

    if pin_type == 'moderator':
        # Extract the radii values for different regions of the moderator pin from the input parameters
        pin_radii = {'moderator': params['Moderator Pin Radii'][0],
                     'cladding': params['Moderator Pin Radii'][1]
        }
        region_keys = ['moderator', 'cladding', 'coolant']

    elif pin_type == 'fuel':
        # Extract the radii values for different regions of the fuel pin from the input parameters
        pin_radii = {'insert': params['Fuel Pin Radii'][0],
                     'gap1': params['Fuel Pin Radii'][1],
                     'fuel_meat': params['Fuel Pin Radii'][2],
                     'gap2': params['Fuel Pin Radii'][3],
                     'cladding': params['Fuel Pin Radii'][4]
        }
        region_keys = ['insert', 'gap1', 'fuel_meat', 'gap2', 'cladding', 'coolant']
    
    else:
        raise ValueError("Invalid pin type. Must be 'moderator' or 'fuel'.")

    # Creating surfaces
    # Create cylindrical surfaces for each of the specified radii
    shells = [openmc.ZCylinder(r=r) for r in pin_radii.values()]

    # Define the regions within the pin using the created cylindrical surfaces
    regions = {}
    for i, key in enumerate(region_keys[:-1]):
        if i == 0:
            regions[key] = -shells[i]
        else:
            regions[key] = +shells[i-1] & -shells[i]
    regions[region_keys[-1]] = +shells[-1]

    return regions


def create_drums_universe(params, control_drum_absorber_material, control_drum_reflector_material, drum_positions):
    number_of_drums = params['Number of Drums']
    valid_drum_counts = [6, 12, 18, 24, 30, 36]
    if number_of_drums not in valid_drum_counts:
        raise ValueError(f"Number of Drums must be one of {valid_drum_counts}, got {number_of_drums}")

    absorber_thickness = params['Drum Absorber Thickness']

=======
    drum_radius = params['Drum Radius']
    absorber_arc = np.deg2rad(params['Drum Absorber Arc Degrees'])
    rotation_angle = 180
    if params['Shutdown Margin Calc']:
>>>>>>> Stashed changes
        rotation_angle = 0
    else:
        rotation_angle = 180

    cd_inner_shell = openmc.ZCylinder(r=drum_radius - absorber_thickness)
    cd_outer_shell = openmc.ZCylinder(r=drum_radius)
    cutting_plane_1 = openmc.Plane(a=np.sin(absorber_arc/2), b= np.cos(absorber_arc/2))
    cutting_plane_2 = openmc.Plane(a=np.sin(absorber_arc/2), b=-np.cos(absorber_arc/2))

    drum_absorber_region = +cd_inner_shell & -cd_outer_shell & -cutting_plane_1 & -cutting_plane_2
    drum_reflector_region = -cd_outer_shell & ~drum_absorber_region
    drum_outside_region   = +cd_outer_shell

    drum_absorber = openmc.Cell(name='drum_absorber', fill=control_drum_absorber_material, region=drum_absorber_region)
    drum_reflector = openmc.Cell(name='drum_reflector', fill=control_drum_reflector_material, region=drum_reflector_region)
    drum_exterior  = openmc.Cell(name='drum_outside', region=drum_outside_region)
    drum_reference = openmc.Universe(cells=(drum_reflector, drum_absorber, drum_exterior))

    drum_cells = []
    for i, (x, y, face_angle_deg) in enumerate(drum_positions):
        dc = openmc.Cell(name=f'drum_{i}', fill=drum_reference)
        dc.rotation = [0, 0, face_angle_deg + rotation_angle]
        drum_cells.append(dc)

    drums = [openmc.Universe(cells=(dc,)) for dc in drum_cells]
    return drums


def create_assembly_universe(params, fuel_pin_universe, moderator_pin_universe, pin_pitch, reflector_material, outer_coolant_universe):
    """
    Creating the universe of the fuel assembly
    @ In, params, dict, The parameters that are used to "fill in" input files with placeholders.
    @ In, fuel_pin_universe, openmc.universe.Universe,
    @ In, moderator_pin_universe, openmc.universe.Universe,
    @ In, pin_pitch, float, the center-to-center distance between adjacent fuel/moderator pins
    @ In, reflector_material, openmc.material.Material, the material of the outer radial reflector
    @ In, outer_coolant_universe, openmc.universe.Universe, the openmc universe of the coolant in the assembly
    @ out, assembly_universe, openmc.universe.Universe, the fuel assembly universe
    """

    assembly = openmc.HexLattice()
    assembly.center = (0., 0.)
    assembly.pitch = (pin_pitch,)
    assembly.outer = outer_coolant_universe

    rings = copy.deepcopy(params['Pins Arrangement'])
    rings = rings[-params['Number of Rings per Assembly']:]

    for i in range(len(rings)):
        for j in range(len(rings[i])):
            if rings[i][j] == 'FUEL':
                rings[i][j] = fuel_pin_universe
            elif rings[i][j] == 'MODERATOR':
                rings[i][j] = moderator_pin_universe

    assembly.universes = rings

    assembly_boundary = openmc.model.hexagonal_prism(
        edge_length=pin_pitch * (params['Number of Rings per Assembly'] - 1) + pin_pitch * 0.6,
        corner_radius=(params['Fuel Pin Radii'])[-1] + params["Pin Gap Distance"]
    )

    fuel_assembly_cell = openmc.Cell(fill=assembly, region=assembly_boundary)
    reflector_cell = openmc.Cell(fill=reflector_material, region=~assembly_boundary)

    assembly_universe = openmc.Universe(cells=[fuel_assembly_cell, reflector_cell])

    return assembly_universe


def create_control_drums_positions(params):
    """
    Place N/6 drums along each of the 6 flat faces of the hexagonal lattice,
    touching each face from outside, evenly spaced along the face length.
    """
    number_of_drums = params['Number of Drums']
    valid_drum_counts = [6, 12, 18, 24, 30, 36]
    if number_of_drums not in valid_drum_counts:
        raise ValueError(
            f"Number of Drums must be one of {valid_drum_counts}, got {number_of_drums}"
        )

    drums_per_side = number_of_drums // 6

    pin_pitch = 2 * params['Fuel Pin Radii'][-1] + params['Pin Gap Distance']
    hex_edge_length = pin_pitch * (params['Number of Rings per Assembly'] - 1) + pin_pitch * 0.6
    apothem = np.sin(np.pi / 3) * hex_edge_length

    drum_radius = params['Drum Radius']
    drum_tube_radius = drum_radius + drum_radius / 90.0
    side_length = hex_edge_length

    # Same-face neighbor spacing check
    same_face_spacing = side_length / drums_per_side
    if 2.0 * drum_tube_radius > same_face_spacing:
        max_drum_radius = (same_face_spacing / 2.0) * 90.0 / 91.0
        raise ValueError(
            f"Drums on the same hex face will overlap. "
            f"For Number of Drums = {number_of_drums}, the maximum Drum Radius is about "
            f"{max_drum_radius:.3f} cm, but got {drum_radius:.3f} cm."
        )

    face_angles = [k * np.pi / 3 for k in range(6)]

    positions = []
    for face_angle in face_angles:
        along_x = -np.sin(face_angle)
        along_y =  np.cos(face_angle)

        # Drum centers are placed just outside each hex face
        radial_distance = apothem + drum_tube_radius
        face_center_x = radial_distance * np.cos(face_angle)
        face_center_y = radial_distance * np.sin(face_angle)

        for i in range(drums_per_side):
            offset = side_length * (i - (drums_per_side - 1) / 2.0) / drums_per_side
            x = face_center_x + offset * along_x
            y = face_center_y + offset * along_y
            positions.append((x, y, np.degrees(face_angle)))

    # Full pairwise overlap check at the placement stage
    min_center_dist = 2.0 * drum_tube_radius
    for i in range(len(positions)):
        x1, y1, _ = positions[i]
        for j in range(i + 1, len(positions)):
            x2, y2, _ = positions[j]
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dist < min_center_dist:
                overlap = min_center_dist - dist
                raise ValueError(
                    f"Drum placement overlap detected between drums {i} and {j}. "
                    f"Overlap = {overlap:.3f} cm. "
                    f"Try reducing Drum Radius or reducing Number of Drums."
                )

    return positions



def create_core_geometry(params, drums, drums_positions, assembly_universe):
    """
    Build the full 2D radial core geometry with reflector-embedded control drums.
    Includes overlap checks and places each drum at its prescribed position.
    """
    params['Drum Tube Radius'] = params['Drum Radius'] + params['Drum Radius'] / 90.0
    drum_tube_radius = params['Drum Tube Radius']

    # Outer vacuum boundary
    outer_surface = openmc.ZCylinder(r=params['Core Radius'], boundary_type='vacuum')

    # Pairwise overlap check
    for i in range(len(drums_positions)):
        x1, y1, _ = drums_positions[i]
        for j in range(i + 1, len(drums_positions)):
            x2, y2, _ = drums_positions[j]
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dist < 2.0 * drum_tube_radius:
                overlap = 2.0 * drum_tube_radius - dist
                raise ValueError(
                    f"Drums {i} and {j} overlap by {overlap:.3f} cm! "
                    f"Reduce Drum Radius or reduce Number of Drums."
                )

    drum_shells = []
    drum_cells = []

    for (x, y, _), du in zip(drums_positions, drums):
        drum_shell = openmc.ZCylinder(x0=x, y0=y, r=drum_tube_radius)
        drum_shells.append(drum_shell)

        # Limit each drum cell to the outer vacuum boundary too
        drum_cell = openmc.Cell(fill=du, region=-drum_shell & -outer_surface)
        drum_cell.translation = (x, y, 0)
        drum_cells.append(drum_cell)

    drums_outside = +drum_shells[0]
    for d in drum_shells[1:]:
        drums_outside = drums_outside & +d

    core_cell = openmc.Cell(fill=assembly_universe, region=-outer_surface & drums_outside)

    core = openmc.Universe(cells=[core_cell] + drum_cells)
    core_geometry = openmc.Geometry(core)

    return core_geometry, core

# **************************************************************************************************************************
#                                                Sec. 1 : OpenMC Model
# **************************************************************************************************************************

"""
An OpenMC function that accepts an instance of "parameters" 
and generates the necessary XMl files
"""

def build_openmc_model_LTMR(params):
    """
    OpenMC Model
    @ In, params, watts.parameters.Parameters, The parameters that are used to "fill in" 
    input files with placeholders. params mostly behaves like a Python dictionary with a few extra capabilities
    """

    params.setdefault('Shutdown Margin Calc', False)
    params.setdefault('Isothermal Temperature Coefficients', False)
    # **************************************************************************************************************************
    #                                                Sec. 1.1 : MATERIALS
    # **************************************************************************************************************************
   
    # reading all the materials properties
    materials_database = collect_materials_data(params)

    # reading the materials properties for the fuel, coolant, refelctor, control drum (the drum includes two materials: absoerber and reflector)   
    fuel = materials_database[params['Fuel']]
    coolant = materials_database[params['Coolant']]
    reflector = materials_database[params['Radial Reflector']]
    control_drum_absorber = materials_database[params['Control Drum Absorber']]
    control_drum_reflector = materials_database[params['Control Drum Reflector']]
    

    # **************************************************************************************************************************
    #                                                Sec. 1.2 : GEOMETRY: Fuel Pins, Moderator Pins, Coolant
    # **************************************************************************************************************************
    
    # Creating Fuel Pins regions
    fuel_pin_regions = create_pin_regions(params, 'fuel')
    
    # Creating fuel pin materials
    fuel_materials = []
    for mat in params['Fuel Pin Materials']:
        if mat == None:
            fuel_materials.append(None)
        else: 
            material_1 = materials_database[mat]
            fuel_materials.append(material_1)
    fuel_materials.append(coolant)

     # Giving the user error message if the number of materials is not the same as the number of regions
    assert len(fuel_pin_regions) == len(fuel_materials), "The number of regions, {len(fuel_pin_regions)} should be\
        the same as the number of introduced materials, {len(fuel_materials)}"
    
    # creating the fuel pin universe
    fuel_cells = create_cells(fuel_pin_regions, fuel_materials)
    # The fuel region cell (to be used in distribcell tally)
    fuel_cell = fuel_cells['fuel_meat']
    fuel_pin_universe = openmc.Universe(cells=fuel_cells.values())

    if params['plotting'] == "Y":
        # plotting
        create_universe_plot(materials_database, fuel_pin_universe, 
                        plot_width = 2.2 * params['Fuel Pin Radii'][-1],
                        num_pixels = 500, 
                        font_size = 32,
                        title = "Fuel Pin Universe", 
                        fig_size = 8, 
                        output_file_name = "fuel_pin_universe.png")

    
    # Creating Moderator Pin regions
    moderator_pin_regions = create_pin_regions(params, 'moderator')
    
    # Creating moderator pin materials
    moderator_materials = []
    for mat in params['Moderator Pin Materials']:
        if mat == None:
            moderator_materials.append(None)
        else: 
            material_1 = materials_database[mat]
            moderator_materials.append(material_1)
    moderator_materials.append(coolant)
    
    # Giving the user error message if the number of materials is not the same as the number of regions
    assert len(moderator_pin_regions) == len(moderator_materials), "The number of regions, {len(moderator_pin_regions)} should be\
        the same as the number of introduced materials, {len(moderator_materials)}"

    # creating them moerator pin universe
    moderator_cells = create_cells(moderator_pin_regions, moderator_materials)
    moderator_pin_universe = openmc.Universe(cells=moderator_cells.values())

    if params['plotting'] == "Y":
        # plotting
        create_universe_plot(materials_database, moderator_pin_universe, 
                        plot_width = 2.2 * params['Moderator Pin Radii'][-1],
                        num_pixels = 500, 
                        font_size = 32,
                        title = "Moderator Pin Universe", 
                        fig_size = 8, 
                        output_file_name = "moderator_pin_universe.png")
    
    
    # Coolant Universe
    coolant_cell = openmc.Cell(fill=coolant)
    coolant_universe = openmc.Universe(cells=(coolant_cell,))
   
    # **************************************************************************************************************************
    #                                                Sec. 1.3 : CONTROL DRUMS
    # **************************************************************************************************************************
    
    
    # **************************************************************************************************************************
    #                                                Sec. 1.4 : Fuel Assembly
    # **************************************************************************************************************************
    
    # The center-to-center distance between adjacent fuel/moderator pins
    pin_pitch = 2 * (params['Fuel Pin Radii'][-1] ) + params["Pin Gap Distance"]

    assembly_universe = create_assembly_universe(params,
                                                 fuel_pin_universe,
                                                 moderator_pin_universe,
                                                 pin_pitch,
                                                 reflector,
                                                 coolant_universe)


    # # **************************************************************************************************************************
    # #                                                Sec. 1.5 : VOLUME INFO for Depletion
    # # **************************************************************************************************************************
    #find where the fuel is in the fuel pin
    fuel_index = params['Fuel Pin Materials'].index(params['Fuel'])

    fissile_area = circle_area(params['Fuel Pin Radii'][fuel_index] )\
        - circle_area(params['Fuel Pin Radii'][fuel_index - 1])
    fuel.volume = fissile_area *params['Active Height'] * params['Fuel Pin Count']
   
    all_materials = fuel_materials +\
        moderator_materials + [coolant, reflector, control_drum_absorber, control_drum_reflector]
    
    # removing "None" materials
    all_materials_cleaned_list = [item for item in all_materials if item is not None]
    materials = openmc.Materials(list(set(all_materials_cleaned_list)))
   
    openmc.Materials.cross_sections = params['cross_sections_xml_location']
    materials.export_to_xml()
    

    # # **************************************************************************************************************************
    # #                                                Sec. 1.6 : CORE DRUM REPLACEMENT
    # # **************************************************************************************************************************

    control_drum_positions = create_control_drums_positions(params)
    drums = create_drums_universe(params, control_drum_absorber, control_drum_reflector, control_drum_positions)

    core_geometry, core = create_core_geometry(params,
                                         drums,
                                         drums_positions = control_drum_positions,
                                         assembly_universe  = assembly_universe )


    core_geometry.export_to_xml()
    
    if params['plotting'] == "Y":
        create_universe_plot(materials_database, core_geometry, 
                        plot_width = 2.01 * params['Core Radius'],
                        num_pixels = 2000, 
                        font_size = 32,
                        title = "Reactor Core", 
                        fig_size = 8, 
                        output_file_name = "core.png")
    
    # # **************************************************************************************************************************
    # #                                                Sec. 1.7 : TALLIES
    # # **************************************************************************************************************************

    tallies_file = openmc.Tallies()

    group_edges = np.array([1e-5, 6.7e-2, 3.2e-1, 1, 4, 9.88, 4.81e1, 4.54e2, 4.9e4, 1.83e5, 8.21e5, 4e7])# 11 energy groups from HPMR report table no.5 in ev
    groups = openmc.mgxs.EnergyGroups(group_edges)

    mgxs_lib = openmc.mgxs.Library(core_geometry)
    mgxs_lib.energy_groups = groups
    mgxs_lib.legendre_order     = 1
    mgxs_lib.mgxs_types = ['absorption', 'diffusion-coefficient', 'transport', 'scatter matrix', 'total', 'scatter']
    mgxs_lib.domain_type = 'universe'
    mgxs_lib.domains = [core]
    mgxs_lib.build_library()
    mgxs_lib.add_to_tallies_file(tallies_file, merge=False)

    # Peaking factor tally (pin power)
    pin_filter = openmc.DistribcellFilter(fuel_cell)
    pin_power = openmc.Tally(name='pin_power_kappa')
    pin_power.scores = ['kappa-fission']
    pin_power.filters = [pin_filter]
    tallies_file.append(pin_power)
    tallies_file.export_to_xml()

    # # **************************************************************************************************************************
    # #                                                Sec. 1.8 : SIMULATION
    # # **************************************************************************************************************************
 
    point = openmc.stats.Point((0, 0, 0))
    source = openmc.Source(space=point)
    settings = openmc.Settings()
    settings.source = source
    settings.batches = 100
    settings.inactive = 50
    if 'Particles' in params.keys():
        settings.particles = int(params['Particles'])#1000
    else:
        settings.particles = 2000 
    if params['Isothermal Temperature Coefficients']:
        settings.temperature = {'default': params['Common Temperature'],
                                 'method': 'interpolation',
                                 'tolerance': 50.0}
    else:
        settings.temperature = {'method': 'interpolation'}  # added missing else branch
    
    settings.export_to_xml()