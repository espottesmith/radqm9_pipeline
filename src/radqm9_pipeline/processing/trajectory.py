import collections
import itertools
from itertools import chain
from glob import glob
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from monty.serialization import loadfn, dumpfn

from tqdm import tqdm
import h5py
import ast

import ase

import networkx as nx

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, CovalentBondNN
from pymatgen.util.graph_hashing import weisfeiler_lehman_graph_hash

from radqm9_pipeline.elements import read_elements
from radqm9_pipeline.modules import merge_data

from maggma.stores.mongolike import MongoStore


# def filter_features(data: list):
#     dataset = []
#     for item in tqdm(data):
#         formatted_data={}
#         formatted_data['mol_id'] = item['molecule_id']
#         formatted_data['species'] = item['species']
#         formatted_data['charge'] = item['charge'] 
#         formatted_data['spin'] = item['spin_multiplicity']
#         formatted_data['geometries'] = item['geometries']
#         formatted_data['energy'] = item['energies']
#         formatted_data['gradients'] = item['forces']
#         formatted_data['mulliken_partial_charges'] = item['mulliken_partial_charges']
#         formatted_data['mulliken_partial_spins'] = item['mulliken_partial_spins']
#         formatted_data['resp_partial_charges'] = item['resp_partial_charges']
#         formatted_data['dipole_moments'] = item['dipole_moments']
#         formatted_data['resp_dipole_moments'] = item['resp_dipole_moments']
#         dataset.append(formatted_data)
#     return dataset


# def resolve_trajectories(data: list):
#     resolved_data = []
#     unequal_data = []
#     bad_data = []
#     non_spin_broken = []
#     for item in tqdm(data):
#         try:
#             feat = [len(item['geometries']), len(item['gradients']), len(item['energy'])]

#             feat_set = set(feat)
#             if len(feat_set) !=1 :
#                 unequal_data.append([item, feat])
#             else:
#                 len_geo = len(item['geometries'])
#                 if len_geo==1:
#                     resolved_data.append(item)
#                 elif len_geo > 1:
#                     item['geometries'] = list(itertools.chain.from_iterable(item['geometries']))
#                     item['gradients'] = list(itertools.chain.from_iterable(item['gradients']))
#                     item['energy'] = list(itertools.chain.from_iterable(item['energy']))
#                     item['mulliken_partial_charges'] = list(itertools.chain.from_iterable(item['mulliken_partial_charges']))
#                     item['resp_partial_charges'] = list(itertools.chain.from_iterable(item['resp_partial_charges']))
               
#                     try:
#                         item['mulliken_partial_spins'] = list(itertools.chain.from_iterable(item['mulliken_partial_spins']))
#                     except TypeError:
#                         # We will resolve weird spin data later in the pipeline
#                         pass

#                     item['dipole_moments'] = list(itertools.chain.from_iterable(item['dipole_moments']))
#                     item['resp_dipole_moments'] = list(itertools.chain.from_iterable(item['resp_dipole_moments']))

#                     resolved_data.append(item)
#                 else:
#                     bad_data.append(item)
#         except TypeError:
#             non_spin_broken.append(item)
        
#     return resolved_data, unequal_data, bad_data, non_spin_broken


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
        item['energy'] = [x*27.2114 for x in energy]

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
            res = np.sqrt(sum([j**2 for j in atom]))
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
                calc_dipole_moments_resp = [pair['calc_resp_dipole_moments'][0], pair['calc_resp_dipole_moments'][-1]]
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
                calc_dipole_moments_resp = [pair['calc_resp_dipole_moments'][0], pair['calc_resp_dipole_moments'][max_index], pair['calc_resp_dipole_moments'][-1]]

            del pair["energy"]

            pair['geometries'] = geometries
            pair['energies'] = energies
            pair['gradients'] = grads
            pair['mulliken_partial_charges'] = mulliken_partial_charges
            pair['mulliken_partial_spins'] = mulliken_partial_spins
            pair['resp_partial_charges'] = resp_partial_charges
            pair['dipole_moments'] = dipole_moments
            pair['resp_dipole_moments'] = resp_dipole_moments
            pair['calc_resp_dipole_moments'] = calc_dipole_moments_resp

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
        
    Both "energy" and "forces" are the dict strings in data.
    """
    atom_list = []
    for i in range(len(data['geometries'])):
        atoms = ase.atoms.Atoms(
            symbols=data['species'],
            positions=data['geometries'][i]
        )
        atoms.arrays['mulliken_partial_charges']=np.array(data['mulliken_partial_charges'][i])
        atoms.arrays['mulliken_partial_spins']=np.array(data['mulliken_partial_spins'][i])
        atoms.arrays['resp_partial_charges']=np.array(data['resp_partial_charges'][i])
        atoms.info['dipole_moments'] = np.array(data['dipole_moments'][i])
        atoms.info['resp_dipole_moments'] = np.array(data['resp_dipole_moments'][i])
        atoms.info['calc_resp_dipole_moments']=np.array(data['calc_resp_dipole_moments'][i])
        atoms.info['weight'] = data['weight']
        
        if energy is not None:
            atoms.info['energy'] = data[energy][i]
        if forces is not None:
            atoms.arrays['forces'] = np.array(data[forces][i])
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


def build_atoms_minimal(data: dict,
                energy: str = None,
                forces: str = None,
                charge:str = None,
                spin:str = None
                ) -> ase.Atoms:
    """ 
    Populate Atoms class with atoms in molecule.
        atoms.info : global variables
        atoms.array : variables for individual atoms
        
    Both "energy" and "forces" are the dict strings in data.
    """
    atom_list = []
    for i in range(len(data['geometries'])):
        atoms = ase.atoms.Atoms(
            symbols=data['species'],
            positions=data['geometries'][i]
        )
        
        atoms.info['weight'] = data['weight']

        if energy is not None:
            atoms.info['energy'] = data[energy][i]
        if forces is not None:
            atoms.arrays['forces'] = np.array(data[forces][i])
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


def build_minimal_atoms_iterator(
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
        atoms=build_atoms_minimal(point, energy=energy, forces=forces, charge=charge, spin=spin)
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

    base_path = ""
    full_data_path = os.path.join(base_path, "full")
    minimal_data_path = os.path.join(base_path, "minimal")

    elements_dict = read_elements('/global/cfs/projectdirs/matgen/ewcss/radqm9/radqm9_pipeline/src/radqm9_pipeline/elements/elements.pkl')

    # Trajectory information
    traj_store = MongoStore(database="thermo_chem_storage",
                            collection_name="radqm9_trajectories",
                            username="thermo_chem_storage_ro",
                            password="",
                            host="mongodb07.nersc.gov",
                            port=27017,
                            key="molecule_id")
    traj_store.connect()

    r_data = []
    for entry in traj_store.query(
        {},
        {
            "molecule_id": 1,
            "species": 1,
            "charge": 1,
            "spin_multiplicity": 1,
            "geometries": 1,
            "energies": 1,
            "forces": 1,
            "mulliken_partial_charges": 1,
            "mulliken_partial_spins": 1,
            "resp_partial_charges": 1,
            "dipole_moments": 1,
            "resp_dipole_moments": 1
        }
    ):
        item = {}
        item['mol_id'] = entry['molecule_id']
        item['species'] = entry['species']
        item['charge'] = entry['charge'] 
        item['spin'] = entry['spin_multiplicity']
        item['geometries'] = entry['geometries']
        item['energy'] = entry['energies']
        item['gradients'] = entry['forces']
        item['mulliken_partial_charges'] = entry['mulliken_partial_charges']
        item['mulliken_partial_spins'] = entry['mulliken_partial_spins']
        item['resp_partial_charges'] = entry['resp_partial_charges']
        item['dipole_moments'] = entry['dipole_moments']
        item['resp_dipole_moments'] = entry['resp_dipole_moments']

        try:
            feat = [len(item['geometries']), len(item['gradients']), len(item['energy'])]

            feat_set = set(feat)
            if len(feat_set) !=1 :
                continue
            else:
                len_geo = len(item['geometries'])
                if len_geo==1:
                    r_data.append(item)
                elif len_geo > 1:
                    item['geometries'] = list(itertools.chain.from_iterable(item['geometries']))
                    item['gradients'] = list(itertools.chain.from_iterable(item['gradients']))
                    item['energy'] = list(itertools.chain.from_iterable(item['energy']))
                    item['mulliken_partial_charges'] = list(itertools.chain.from_iterable(item['mulliken_partial_charges']))
                    item['resp_partial_charges'] = list(itertools.chain.from_iterable(item['resp_partial_charges']))
               
                    try:
                        item['mulliken_partial_spins'] = list(itertools.chain.from_iterable(item['mulliken_partial_spins']))
                    except TypeError:
                        # We will resolve weird spin data later in the pipeline
                        pass

                    item['dipole_moments'] = list(itertools.chain.from_iterable(item['dipole_moments']))
                    item['resp_dipole_moments'] = list(itertools.chain.from_iterable(item['resp_dipole_moments']))

                    r_data.append(item)
                else:
                    continue
        except TypeError:
            continue

    add_unique_id(r_data)

    dimension(r_data)

    pc = filter_field(r_data, 'mulliken_partial_charges')
    ps = filter_field(r_data, 'mulliken_partial_spins')
    rpc = filter_field(r_data, 'resp_partial_charges')
    dm = filter_field(r_data, 'dipole_moments')
    rdm = filter_field(r_data, 'resp_dipole_moments')

    missing_data = {
        "mulliken_partial_charges": {x: len(pc[x]) for x in pc.keys()},
        "mulliken_partial_spins": {x: len(ps[x]) for x in ps.keys()},
        "resp_partial_charges": {x: len(rpc[x]) for x in rpc.keys()},
        "dipole_moments": {x: len(dm[x]) for x in dm.keys()},
        "resp_dipole_moments": {x: len(rdm[x]) for x in rdm.keys()},
    }

    dumpfn(missing_data, os.path.join(base_path, "missing_data.json"))

    generate_resp_dipole(r_data)

    resolve_mulliken_partial_spins(r_data)

    dumpfn(r_data, os.path.join(base_path, "raw_trajectory_data.json"))

    r_data = filter_data(r_data)

    r_data = force_magnitude_filter(cutoff=10.0, data=r_data)

    convert_energy_forces(r_data)

    molecule_weight(r_data, elements_dict)

    traj_all = build_atoms_iterator(r_data)

    file = os.path.join(base_path, 'radqm9_65_10_25_trajectory_data_20240807_all.xyz')
    ase.io.write(file, traj_all, format="extxyz")

    # Cleaning for memory
    del traj_all

    sparse_trajectory(r_data)

    dumpfn(r_data, os.path.join(base_path, "clean_trajectory_data.json"))

    r_data, ood = filter_broken_graphs(r_data)

    wtd = weight_to_data(r_data)
    sld = length_dict(wtd)

    # Cleaning for memory
    del r_data

    train_mass = ['152.037'] # EVAN WILL NEED TO ADJUST THE MASSES OF INITIAL POINTS FOR NEW DATA
    test_mass = ['144.092']
    val_mass = ['143.108']

    train = sld['152.037'] # trackers for dataset sizes
    test = sld['144.092']
    val = sld['143.108']

    sld.pop('152.037')
    sld.pop('144.092')
    sld.pop('143.108')

    # Sort the data 
    # data is a dict: mass-># of trajs
    for mass in sld:
        temp_total = train + val + test
        train_ratio = .65 - (train / temp_total)
        test_ratio = .25 - (test / temp_total)
        val_ratio = .1 - (val / temp_total)
        
        if train_ratio > val_ratio and train_ratio>test_ratio:
            train_mass.append(mass)
            train += sld[mass]
        elif val_ratio > train_ratio and val_ratio>test_ratio:
            val_mass.append(mass)
            val += sld[mass]
        elif test_ratio > val_ratio and test_ratio>train_ratio:
            test_mass.append(mass)
            test += sld[mass]

    sld = length_dict(wtd) # you need to call this again yes

    data = {
        "train": list(),
        "val": list(),
        "test": list()
    }

    for split, masses in [("train", train_mass), ("val", val_mass), ("test", test_mass)]:
        for mass in masses:
            for mpoint in wtd[mass]:
                data[split].append(mpoint)

    # Minimal build
    build_minimal = dict()
    for split in data:
        build_minimal[split] = build_minimal_atoms_iterator(data[split], energy="energies")
        
    create_dataset(build_minimal, 'radqm9_65_10_25_trajectory_minimal_data_20240807', minimal_data_path)

    ood_minimal = build_minimal_atoms_iterator(ood, energy="energies")
    file = os.path.join(minimal_data_path, 'radqm9_65_10_25_trajectory_minimal_data_20240807_ood.xyz')
    ase.io.write(file, ood_minimal,format="extxyz")

    # Charge/spin subsets
    train_cs_dict = {}
    for item in tqdm(build_minimal['train']):
        key = str(item.info['charge'])+str(item.info['spin'])
        try:
            train_cs_dict[key].append(item)
        except KeyError:
            train_cs_dict[key] = [item]

    val_cs_dict = {}
    for item in tqdm(build_minimal['val']):
        key = str(item.info['charge'])+str(item.info['spin'])
        try:
            val_cs_dict[key].append(item)
        except KeyError:
            val_cs_dict[key] = [item]

    test_cs_dict = {}
    for item in tqdm(build_minimal['test']):
        key = str(item.info['charge'])+str(item.info['spin'])
        try:
            test_cs_dict[key].append(item)
        except KeyError:
            test_cs_dict[key] = [item]

    ood_cs_dict = {}
    for item in tqdm(ood_minimal):
        key = str(item.info['charge']) + str(item.info['spin'])
        try:
            ood_cs_dict[key].append(item)
        except KeyError:
            ood_cs_dict[key] = [item]

    # Split by charge/spin pair
    # Use this for relative energies
    minimal_chargespin_path = os.path.join(minimal_data_path, "by_charge_spin")
    if not os.path.exists(minimal_chargespin_path):
        os.mkdir(minimal_chargespin_path)

    for key in test_cs_dict:
        file = os.path.join(minimal_chargespin_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_train_'+key+'.xyz')
        ase.io.write(file, train_cs_dict[key], format="extxyz")
        
        file = os.path.join(minimal_chargespin_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_val_'+key+'.xyz')
        ase.io.write(file, val_cs_dict[key],format="extxyz")
        
        file = os.path.join(minimal_chargespin_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_test_'+key+'.xyz')
        ase.io.write(file, test_cs_dict[key],format="extxyz")

        if key in ood_cs_dict:
            file = os.path.join(minimal_chargespin_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_ood_'+key+'.xyz')
            ase.io.write(file, ood_cs_dict[key],format="extxyz")

    # Doublet
    minimal_doublet_path = os.path.join(minimal_data_path, "doublet")
    if not os.path.exists(minimal_doublet_path):
        os.mkdir(minimal_doublet_path)

    doublet_train = []
    doublet_val = []
    doublet_test = []
    doublet_ood = []

    for item in tqdm(build_minimal['train']):
        if item.info['spin'] == 2:
            doublet_train.append(item)

    for item in tqdm(build_minimal['val']):
        if item.info['spin'] == 2:
            doublet_val.append(item)

    for item in tqdm(build_minimal['test']):
        if item.info['spin'] == 2:
            doublet_test.append(item)

    for item in tqdm(ood_minimal):
        if item.info['spin'] == 2:
            doublet_ood.append(item)

    file = os.path.join(minimal_doublet_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_doublet_train.xyz')
    ase.io.write(file, doublet_train, format="extxyz")
    
    file = os.path.join(minimal_doublet_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_doublet_val.xyz')
    ase.io.write(file, doublet_val, format="extxyz")
    
    file = os.path.join(minimal_doublet_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_doublet_test.xyz')
    ase.io.write(file, doublet_test, format="extxyz")

    file = os.path.join(minimal_doublet_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_doublet_ood.xyz')
    ase.io.write(file, doublet_ood, format="extxyz")


    # Neutral
    minimal_neutral_path = os.path.join(minimal_data_path, "neutral")
    if not os.path.exists(minimal_neutral_path):
        os.mkdir(minimal_neutral_path)

    neutral_train = []
    neutral_val = []
    neutral_test = []
    neutral_ood = []

    for item in tqdm(build_minimal['train']):
        if item.info['charge'] == 0:
            neutral_train.append(item)

    for item in tqdm(build_minimal['val']):
        if item.info['charge'] == 0:
            neutral_val.append(item)

    for item in tqdm(build_minimal['test']):
        if item.info['charge'] == 0:
            neutral_test.append(item)

    for item in tqdm(ood_minimal):
        if item.info['charge'] == 0:
            neutral_ood.append(item)

    file = os.path.join(minimal_neutral_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_neutral_train.xyz')
    ase.io.write(file, neutral_train, format="extxyz")
    
    file = os.path.join(minimal_neutral_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_neutral_val.xyz')
    ase.io.write(file, neutral_val, format="extxyz")
    
    file = os.path.join(minimal_neutral_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_neutral_test.xyz')
    ase.io.write(file, neutral_test, format="extxyz")

    file = os.path.join(minimal_neutral_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_neutral_ood.xyz')
    ase.io.write(file, neutral_ood, format="extxyz")

    fractions = [.01, .05, .1, .25, .5, .75]

    wtd_minimal = weight_to_data_ase(build_minimal["train"])
    cd_minimal = chunk_data(wtd_minimal, fractions)
    
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
        chunk_file = os.path.join(minimal_data_path, 'radqm9_65_10_25_trajectory_minimal_data_20240807_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_minimal[ii],format="extxyz")
        
        for key in cd_cs:
            chunk_file = os.path.join(minimal_chargespin_path, 'radqm9_65_10_25_trajectory_minimal_data_20240807_train_subset_' + key + f'_{frac}.xyz')
            ase.io.write(chunk_file, cd_cs[key][ii], format="extxyz")
            
        chunk_file = os.path.join(minimal_doublet_path, 'radqm9_65_10_25_trajectory_minimal_data_20240807_doublet_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_doublet[ii],format="extxyz")
        
        chunk_file = os.path.join(minimal_neutral_path, 'radqm9_65_10_25_trajectory_minimal_data_20240807_neutral_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_neutral[ii],format="extxyz")

    # Cleanup for memory
    del build_minimal
    del ood_minimal
    del train_cs_dict
    del val_cs_dict
    del test_cs_dict
    del ood_cs_dict
    del doublet_train
    del doublet_val
    del doublet_test
    del doublet_ood
    del neutral_train
    del neutral_val
    del neutral_test
    del neutral_ood
    del wtd_minimal
    del wtd_cs
    del wtd_doublet
    del wtd_neutral

    # Full build
    build_full = dict()
    for split in data:
        build_full[split] = build_atoms_iterator(data[split], energy="energies")
        
    create_dataset(build_full, 'radqm9_65_10_25_trajectory_full_data_20240807', full_data_path)

    ood_full = build_minimal_atoms_iterator(ood, energy="energies")
    file = os.path.join(full_data_path, 'radqm9_65_10_25_trajectory_full_data_20240807_ood.xyz')
    ase.io.write(file, ood_full, format="extxyz")

    # Charge/spin subsets
    train_cs_dict = {}
    for item in tqdm(build_full['train']):
        key = str(item.info['charge'])+str(item.info['spin'])
        try:
            train_cs_dict[key].append(item)
        except KeyError:
            train_cs_dict[key] = [item]

    val_cs_dict = {}
    for item in tqdm(build_full['val']):
        key = str(item.info['charge'])+str(item.info['spin'])
        try:
            val_cs_dict[key].append(item)
        except KeyError:
            val_cs_dict[key] = [item]

    test_cs_dict = {}
    for item in tqdm(build_full['test']):
        key = str(item.info['charge'])+str(item.info['spin'])
        try:
            test_cs_dict[key].append(item)
        except KeyError:
            test_cs_dict[key] = [item]

    ood_cs_dict = {}
    for item in tqdm(ood_full):
        key = str(item.info['charge']) + str(item.info['spin'])
        try:
            ood_cs_dict[key].append(item)
        except KeyError:
            ood_cs_dict[key] = [item]

    # Split by charge/spin pair
    # Use this for relative energies
    full_chargespin_path = os.path.join(full_data_path, "by_charge_spin")
    if not os.path.exists(full_chargespin_path):
        os.mkdir(full_chargespin_path)

    for key in test_cs_dict:
        file = os.path.join(full_chargespin_path,'radqm9_65_10_25_trajectory_full_data_20240807_train_'+key+'.xyz')
        ase.io.write(file, train_cs_dict[key], format="extxyz")
        
        file = os.path.join(full_chargespin_path,'radqm9_65_10_25_trajectory_full_data_20240807_val_'+key+'.xyz')
        ase.io.write(file, val_cs_dict[key],format="extxyz")
        
        file = os.path.join(full_chargespin_path,'radqm9_65_10_25_trajectory_full_data_20240807_test_'+key+'.xyz')
        ase.io.write(file, test_cs_dict[key],format="extxyz")

        if key in ood_cs_dict:
            file = os.path.join(full_chargespin_path,'radqm9_65_10_25_trajectory_full_data_20240807_ood_'+key+'.xyz')
            ase.io.write(file, ood_cs_dict[key], format="extxyz")

    # Doublet
    full_doublet_path = os.path.join(full_data_path, "doublet")
    if not os.path.exists(full_doublet_path):
        os.mkdir(full_doublet_path)

    doublet_train = []
    doublet_val = []
    doublet_test = []
    doublet_ood = []

    for item in tqdm(build_full['train']):
        if item.info['spin'] == 2:
            doublet_train.append(item)

    for item in tqdm(build_full['val']):
        if item.info['spin'] == 2:
            doublet_val.append(item)

    for item in tqdm(build_full['test']):
        if item.info['spin'] == 2:
            doublet_test.append(item)

    for item in tqdm(ood_full):
        if item.info['spin'] == 2:
            doublet_ood.append(item)

    file = os.path.join(full_doublet_path,'radqm9_65_10_25_trajectory_full_data_20240807_doublet_train.xyz')
    ase.io.write(file, doublet_train, format="extxyz")
    
    file = os.path.join(full_doublet_path,'radqm9_65_10_25_trajectory_full_data_20240807_doublet_val.xyz')
    ase.io.write(file, doublet_val,format="extxyz")
    
    file = os.path.join(full_doublet_path,'radqm9_65_10_25_trajectory_full_data_20240807_doublet_test.xyz')
    ase.io.write(file, doublet_test,format="extxyz")

    file = os.path.join(full_doublet_path,'radqm9_65_10_25_trajectory_full_data_20240807_doublet_ood.xyz')
    ase.io.write(file, doublet_ood, format="extxyz")

    # Neutral
    full_neutral_path = os.path.join(full_data_path, "neutral")
    if not os.path.exists(full_neutral_path):
        os.mkdir(full_neutral_path)

    neutral_train = []
    neutral_val = []
    neutral_test = []
    neutral_ood = []

    for item in tqdm(build_full['train']):
        if item.info['charge'] == 0:
            neutral_train.append(item)

    for item in tqdm(build_full['val']):
        if item.info['charge'] == 0:
            neutral_val.append(item)

    for item in tqdm(build_full['test']):
        if item.info['charge'] == 0:
            neutral_test.append(item)

    for item in tqdm(ood_full):
        if item.info['charge'] == 0:
            neutral_ood.append(item)

    file = os.path.join(full_neutral_path,'radqm9_65_10_25_trajectory_full_data_20240807_neutral_train.xyz')
    ase.io.write(file, neutral_train, format="extxyz")
    
    file = os.path.join(full_neutral_path,'radqm9_65_10_25_trajectory_full_data_20240807_neutral_val.xyz')
    ase.io.write(file, neutral_val, format="extxyz")
    
    file = os.path.join(full_neutral_path,'radqm9_65_10_25_trajectory_full_data_20240807_neutral_test.xyz')
    ase.io.write(file, neutral_test, format="extxyz")

    file = os.path.join(full_neutral_path,'radqm9_65_10_25_trajectory_full_data_20240807_neutral_ood.xyz')
    ase.io.write(file, neutral_ood, format="extxyz")

    fractions = [.01, .05, .1, .25, .5, .75]

    wtd_full = weight_to_data_ase(build_full["train"])
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
        chunk_file = os.path.join(full_data_path, 'radqm9_65_10_25_trajectory_full_data_20240807_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_minimal[ii],format="extxyz")
        
        for key in cd_cs:
            chunk_file = os.path.join(full_chargespin_path, 'radqm9_65_10_25_trajectory_full_data_20240807_train_subset_' + key + f'_{frac}.xyz')
            ase.io.write(chunk_file, cd_cs[key][ii], format="extxyz")
            
        chunk_file = os.path.join(full_doublet_path, 'radqm9_65_10_25_trajectory_full_data_20240807_doublet_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_doublet[ii],format="extxyz")
        
        chunk_file = os.path.join(full_neutral_path, 'radqm9_65_10_25_trajectory_full_data_20240807_neutral_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_neutral[ii],format="extxyz")

    # TODO: 
    # - Calculate relative energies & regenerate