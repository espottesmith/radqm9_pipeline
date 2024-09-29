import itertools
import math
import os
import random

import numpy as np
from monty.serialization import dumpfn

from tqdm import tqdm

import ase

import networkx as nx

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

from radqm9_pipeline.elements import read_elements

from maggma.stores.mongolike import MongoStore


def add_unique_id(data: list):
    for item in tqdm(data):
        item['charge_spin'] = str(item['charge']) + "_" + str(item['spin'])
        item['mol_cs'] = str(item['mol_id']) + str(item['charge_spin'])


def dimension(data: list):
    fields = []
    for item in tqdm(data): 
        if len(item['geometries']) == 1:
            item['geometries'] = item['geometries'][0]
        if len(item['gradients']) == 1:
            item['gradients'] = item['gradients'][0]
        if len(item['energy']) == 1:
            item['energy'] = item['energy'][0]
        if len(item['mulliken_partial_charges']) == 1:
            item['mulliken_partial_charges'] = item['mulliken_partial_charges'][0]
        if len(item['mulliken_partial_spins']) == 1:
            item['mulliken_partial_spins'] = item['mulliken_partial_spins'][0]
        if len(item['resp_partial_charges']) == 1:
            item['resp_partial_charges'] = item['resp_partial_charges'][0]
        if len(item['dipole_moments']) == 1:
            item['dipole_moments'] = item['dipole_moments'][0]
        if len(item['resp_dipole_moments']) == 1:
            item['resp_dipole_moments'] = item['resp_dipole_moments'][0]


def flatten_list(input_list):
    flattened_list = []
    for item in input_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def filter_field(data, field):
    shit={}
    for item in tqdm(data):
        if item[field] is None:
            try:
                shit[item['charge_spin']].append(item)
            except KeyError:
                shit[item['charge_spin']] = [item]
        elif None in flatten_list(item[field]):
            try:
                shit[item['charge_spin']].append(item)
            except KeyError:
                shit[item['charge_spin']] = [item]
    return shit


def generate_resp_dipole(data: list): #THIS IS GOOD
    for item in tqdm(data):
        resp_dipole = []
        resp_dipole_conv = []
        for i in range(len(item['resp_partial_charges'])):
            resp_partial_charges = np.array(item['resp_partial_charges'][i])
            geometries = np.array(item['geometries'][i])
            
            # Calculate dipole moment components
            dipole_components = resp_partial_charges[:, np.newaxis] * geometries
            
            # Sum the dipole moment components along axis 0 to get the total dipole moment vector
            dipole_moment = np.sum(dipole_components, axis=0)
            dipole_moment_conv = np.sum(dipole_components, axis=0)*(1/0.2081943)
            
            # Append dipole moment to resp_dipole list
            resp_dipole.append(dipole_moment.tolist())  # Convert numpy array to list
            # Append dipole moment to resp_dipole list
            resp_dipole_conv.append(dipole_moment_conv.tolist())  # Convert numpy array to list
        
        item['calc_resp_dipole_moments'] = resp_dipole_conv


def resolve_mulliken_partial_spins(data: list):
    for item in tqdm(data):
        if item['charge_spin']=='0_1':
            if item['mulliken_partial_spins'] is None or None in item['mulliken_partial_spins']:
                charge_array = np.array(item['mulliken_partial_charges'])
                item['mulliken_partial_spins'] = np.zeros(charge_array.shape, dtype=float).tolist()


def filter_data(data: list):
    good = []
    # filtered = []
    for item in data:
        if item['charge_spin'] != '0_1':
            if len(item['gradients']) < 2:
                # filtered.append(item)
                continue
            else:
                good.append(item)
        else:
            good.append(item)
    
    # return good, filtered
    return good


def force_magnitude_filter(cutoff: float,
                           data: list):
    """
    This method returns both data that meets the cuttoff value and data that is equal to or above the cuttoff value.
    If this is run before downsampling, it removes the entire data point trajectory.
    
    Returns: lists
    """
    good = []
    # bad = []
    for item in tqdm(data):
        forces = item['gradients']
        for path_point in forces:
            next_item = False
            for atom in path_point:
                try:
                    res = np.sqrt(sum([i**2 for i in atom]))
                    if res >= cutoff:
                        # bad.append(item)
                        next_item = True
                        break
                except TypeError:
                    res = np.sqrt(sum([i**2 for i in atom[0]]))
                    if res >= cutoff:
                        # bad.append(item)
                        next_item = True
                        break
            if next_item:
                break
        if not next_item:
            good.append(item)
                            
    # return good, bad
    return good

            
def filter_charges(data: list, charge: list):
    clean = []
    bad = []
    for item in data:
        if item['charge'] not in charge:
            clean.append(item)
        else:
            bad.append(item)
    return clean, bad


def convert_energy_forces(data: list):
    for item in tqdm(data):
        energy = item['energy']
        item['energy'] = [x * 27.2114 for x in energy]

        forces = item['gradients']
        traj_arr = []
        for traj_point in forces:
            atom_arr = []
            for atom in traj_point:
                comp_arr = []
                for component in atom:
                    new_component = component * 51.42208619083232
                    comp_arr.append(new_component)
                atom_arr.append(comp_arr)
            traj_arr.append(atom_arr)
        item['gradients'] = traj_arr


def average_force_trajectory(pair):
    """
    This method will take a specfic spin charge pair. At each point in the optimization trajectory, the 
    """
    forces = {}
    for i in range(len(pair['gradients'])):
        temp = []
        for atom in pair['gradients'][i]:
            res = np.sqrt(sum([j ** 2 for j in atom]))
            temp.append(res)
        forces[i] = np.mean(temp)
    del forces[0]
    return forces


def sparse_trajectory(data: list):
    """
    This takes the cleaned data and will sparsifiy the optimization trajectories. How this is done will depend on the
    charge_spin pair:
    - Neutral Singlet (0,1): First and Last
    - Other: First, Last, and structure with the highest molecular force other than the First.
    
    Note: Molecular Force is just the average of the force magnitudes of each atom in the molecule:
    """
    # bad=[]

    for pair in tqdm(data):
        try:
            if pair['charge_spin'] == '0_1':
                geometries = [pair['geometries'][0], pair['geometries'][-1]]
                energies = [pair['energy'][0], pair['energy'][-1]]
                grads = [pair['gradients'][0], pair['gradients'][-1]]
                mulliken_partial_charges = [pair['mulliken_partial_charges'][0], pair['mulliken_partial_charges'][-1]]
                mulliken_partial_spins = [pair['mulliken_partial_spins'][0], pair['mulliken_partial_spins'][-1]]
                resp_partial_charges = [pair['resp_partial_charges'][0], pair['resp_partial_charges'][-1]]
                dipole_moments = [pair['dipole_moments'][0], pair['dipole_moments'][-1]]
                resp_dipole_moments = [pair['resp_dipole_moments'][0], pair['resp_dipole_moments'][-1]]
                # calc_dipole_moments_resp = [pair['calc_resp_dipole_moments'][0], pair['calc_resp_dipole_moments'][-1]]
            else:
                force_dict = average_force_trajectory(pair)
                max_index = max(force_dict, key=force_dict.get)

                geometries = [pair['geometries'][0], pair['geometries'][max_index], pair['geometries'][-1]]
                energies = [pair['energy'][0], pair['energy'][max_index], pair['energy'][-1]]
                grads = [pair['gradients'][0], pair['gradients'][max_index], pair['gradients'][-1]]
                mulliken_partial_charges = [pair['mulliken_partial_charges'][0], pair['mulliken_partial_charges'][max_index], pair['mulliken_partial_charges'][-1]]
                mulliken_partial_spins = [pair['mulliken_partial_spins'][0], pair['mulliken_partial_spins'][max_index], pair['mulliken_partial_spins'][-1]]
                resp_partial_charges = [pair['resp_partial_charges'][0], pair['resp_partial_charges'][max_index], pair['resp_partial_charges'][-1]]
                dipole_moments = [pair['dipole_moments'][0], pair['dipole_moments'][max_index], pair['dipole_moments'][-1]]
                resp_dipole_moments = [pair['resp_dipole_moments'][0], pair['resp_dipole_moments'][max_index], pair['resp_dipole_moments'][-1]]
                # calc_dipole_moments_resp = [pair['calc_resp_dipole_moments'][0], pair['calc_resp_dipole_moments'][max_index], pair['calc_resp_dipole_moments'][-1]]

            del pair["energy"]

            pair['geometries'] = geometries
            pair['energies'] = energies
            pair['gradients'] = grads
            pair['mulliken_partial_charges'] = mulliken_partial_charges
            pair['mulliken_partial_spins'] = mulliken_partial_spins
            pair['resp_partial_charges'] = resp_partial_charges
            pair['dipole_moments'] = dipole_moments
            pair['resp_dipole_moments'] = resp_dipole_moments
            # pair['calc_resp_dipole_moments'] = calc_dipole_moments_resp

        except ValueError:
            continue
            # bad.append(pair)

    # return bad


def molecule_weight(data: list, elements_dict: dict[str, float]):
    """
    This method takes in data and assigns the mass.
    Python does a weird thing floats e.g., {126.15499999999993, 126.15499999999994}, having this and
    get_molecule_weight gurantees that species that are the same are not being assigned different weights.
    """
    for item in tqdm(data):
        total_mass = sum([elements_dict[x] for x in item["species"]])
        item['weight'] = f"{round(total_mass, 3):.3f}"


def weight_to_data(data: list):
    """
    This method buckets the data by the mass such that the dict key is the mass and the values are the data
    points.
    """
    dict_data = {}
    for item in tqdm(data):
        try:
            dict_data[item['weight']].append(item)
        except KeyError:
            dict_data[item['weight']] = [item]
    return dict_data


def length_dict(data: dict):
    """
    This method takes in the output of weight_to_data and returns a dictionary that is sorted from largest
    to smallest mass. The keys are the mass and the values are the number of appearances.
    """
    length_dict = {key: len(value) for key, value in data.items()}
    sorted_length_dict = {k: length_dict[k] for k in sorted(length_dict, key=lambda x: float(x), reverse=True)}
    
    return sorted_length_dict


def build_atoms(data: dict,
                energy: str = None,
                forces: str = None,
                charge:str = None,
                spin:str = None
                ) -> ase.Atoms:
    """ 
    Populate Atoms class with atoms in molecule.
        atoms.info : global variables
        atoms.array : variables for individual atoms
        
    """
    atom_list = []
    for i in range(len(data['geometries'])):
        atoms = ase.atoms.Atoms(
            symbols=data['species'],
            positions=data['geometries'][i]
        )
        atoms.arrays['mulliken_partial_charges'] = np.array(data['mulliken_partial_charges'][i])
        atoms.arrays['mulliken_partial_spins'] = np.array(data['mulliken_partial_spins'][i])
        atoms.arrays['resp_partial_charges'] = np.array(data['resp_partial_charges'][i])
        atoms.info['dipole_moments'] = np.array(data['dipole_moments'][i])
        atoms.info['resp_dipole_moments'] = np.array(data['resp_dipole_moments'][i])
        # atoms.info['calc_resp_dipole_moments'] = np.array(data['calc_resp_dipole_moments'][i])
        atoms.info['weight'] = data['weight']
        
        if energy is not None:
            atoms.info['REF_energy'] = data[energy][i]
        if forces is not None:
            atoms.arrays['REF_forces'] = np.array(data[forces][i])
        if charge is not None:
             atoms.info['charge'] = data[charge]
        if spin is not None:
            atoms.info['spin'] = data[spin]
        atoms.info['mol_id'] = data['mol_id']
        if i == 0:
            atoms.info['position_type'] = 'start'
        if i == 1:
            if data['charge_spin'] == '0_1':
                atoms.info['position_type'] = 'end'
            else:
                atoms.info['position_type'] = 'middle'
        if i == 2:
            atoms.info['position_type'] = 'end'
        atom_list.append(atoms)
    return atom_list


def build_atoms_iterator(
    data: list,
    energy: str = "energy",
    forces: str = "gradients",
    charge:str = "charge",
    spin:str = "spin"
):
    """
    This method assumes the data has been validated. This will create ASE atoms to be written.
    
    The input needs to be a list of lists that contain the event dictionaries. Each inner list needs to represent all the events for a single
    mol_id.
    """
    data_set=[]
    for point in tqdm(data):
        atoms=build_atoms(point, energy=energy, forces=forces, charge=charge, spin=spin)
        data_set+=atoms
    return data_set


def create_dataset(data: dict,
                   file_name:str,
                   path:str):
    """
    This method will handle the I/O for writing the data to xyz files to the path provided.
    """

    if not os.path.exists(path):
        os.mkdir(path)

    train_data = data['train']
    val_data = data['val']
    test_data = data['test']
    
    train_file = os.path.join(path,file_name+'_train.xyz')
    ase.io.write(train_file, train_data,format="extxyz")
     
    val_file = os.path.join(path,file_name+'_val.xyz')
    ase.io.write(val_file, val_data,format="extxyz")
    
    test_file = os.path.join(path,file_name+'_test.xyz')
    ase.io.write(test_file, test_data,format="extxyz")


def chunk_data(data: dict, chunks: list):
    return_data = {}
    foo_data = data
    total=0
    for pair in tqdm(data):
        total+=len(data[pair])
    
    sizes = []
    for item in chunks:
        temp_size = round(total*item)
        sizes.append(temp_size)
    
    for i in range(len(chunks)):
        chunk_data = []
        if i==0:
            for key in tqdm(data):
                if len(foo_data[key]) != 0:
                    random.shuffle(foo_data[key])
                    # print(len(foo_data[key]))
                    sample_size = math.floor(chunks[i] * len(foo_data[key]))
                    chunk_data += foo_data[key][:sample_size]
                    foo_data[key] = foo_data[key][sample_size:]
            return_data[i] = chunk_data
        else:
            counter = 0
            for j in range(50):
                if counter < sizes[i]-sizes[i-1]:
                    for key in data:
                        if len(foo_data[key]) != 0:
                            sample_size = math.floor((chunks[i]-chunks[i-1]) * len(foo_data[key]))
                            add_on = foo_data[key][:sample_size]
                            chunk_data += add_on
                            foo_data[key] = foo_data[key][sample_size:]
                            counter += len(add_on)
                            if counter >= sizes[i]-sizes[i-1]:
                                break
                else:
                    break

            return_data[i] = chunk_data + return_data[i-1]
    return return_data


def weight_to_data_ase(data: list):
    dict_data = {}
    for item in tqdm(data):
        try:
            dict_data[item.info['weight']].append(item)
        except KeyError:
            dict_data[item.info['weight']] = [item]
    return dict_data


def build_graph(species, position):
    atoms = ase.atoms.Atoms(symbols=species,
                            positions=position)
    mol = AseAtomsAdaptor.get_molecule(atoms)
    graph = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
    return graph


def filter_broken_graphs(data: list):
    broken = []
    good = []
    
    for item in tqdm(data):
        if item['charge_spin'] == '0_1':
            good.append(item)
        else:
            isbroken = False
            broken_index=[]
            for i in range(len(item['geometries'])):
                graph = build_graph(item['species'], item['geometries'][i])
                connected = nx.is_connected(graph.graph.to_undirected())
                if not connected:
                    isbroken = True
                    broken_index.append(i)
            if not isbroken:
                good.append(item)
            else:
                broken.append(item)
            
            item['broken_index'] = broken_index

    return good, broken


if __name__ == "__main__":

    full_data_path = "/clusterfs/mp/ewcspottesmith/data/radqm9/process/traj/"

    data = ase.io.read(os.path.join(full_data_path, "radqm9_65_10_25_trajectory_full_data_20240916_train.xyz"))

    # Charge/spin subsets
    train_cs_dict = {}
    for item in tqdm(data):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            train_cs_dict[key].append(item)
        except KeyError:
            train_cs_dict[key] = [item]
    
    full_chargespin_path = os.path.join(full_data_path, "by_charge_spin")
    if not os.path.exists(full_chargespin_path):
        os.mkdir(full_chargespin_path)

    # Doublet
    full_doublet_path = os.path.join(full_data_path, "doublet")
    if not os.path.exists(full_doublet_path):
        os.mkdir(full_doublet_path)

    doublet_train = []
    
    for item in tqdm(data):
        if item.info['spin'] == 2:
            doublet_train.append(item)

    # Neutral
    full_neutral_path = os.path.join(full_data_path, "neutral")
    if not os.path.exists(full_neutral_path):
        os.mkdir(full_neutral_path)

    neutral_train = []

    for item in tqdm(build_full['train']):
        if item.info['charge'] == 0:
            neutral_train.append(item)

    fractions = [.01, .05, .1, .25, .5, .75]

    # TODO: worst comes to worst, change this to chunk by weight and charge
    wtd_full = weight_to_data_ase(data)
    cd_full = chunk_data(wtd_full, fractions)
    
    wtd_cs = dict()
    cd_cs = dict()
    for key in train_cs_dict:
        wtd_cs[key] = weight_to_data_ase(train_cs_dict[key])
        cd_cs[key] = chunk_data(wtd_cs[key], fractions)
    
    wtd_doublet = weight_to_data_ase(doublet_train)
    cd_doublet = chunk_data(wtd_doublet, fractions)

    wtd_neutral = weight_to_data_ase(neutral_train)
    cd_neutral = chunk_data(wtd_neutral, fractions)

    for ii, frac in enumerate(fractions):
        chunk_file = os.path.join(full_data_path, 'radqm9_65_10_25_trajectory_full_data_20240918_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_full[ii],format="extxyz")
        
        for key in cd_cs:
            chunk_file = os.path.join(full_chargespin_path, 'radqm9_65_10_25_trajectory_full_data_20240918_train_subset_' + key + f'_{frac}.xyz')
            ase.io.write(chunk_file, cd_cs[key][ii], format="extxyz")
            
        chunk_file = os.path.join(full_doublet_path, 'radqm9_65_10_25_trajectory_full_data_20240918_doublet_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_doublet[ii],format="extxyz")
        
        chunk_file = os.path.join(full_neutral_path, 'radqm9_65_10_25_trajectory_full_data_20240918_neutral_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_neutral[ii],format="extxyz")
