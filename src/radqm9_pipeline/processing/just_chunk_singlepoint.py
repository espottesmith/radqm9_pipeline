from itertools import chain
import math
import os
import random

import numpy as np
from monty.serialization import dumpfn, loadfn
from maggma.stores.mongolike import MongoStore

from tqdm import tqdm

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

import ase

import networkx as nx

from radqm9_pipeline.elements import read_elements


def filter_duplicate_and_missing_data(data: list):
    filtered_data = []
    
    bucket_mol_id = {}
    for item in tqdm(data):
        try:
            bucket_mol_id[item['mol_id']].append(item)
        except KeyError:
            bucket_mol_id[item['mol_id']] = [item]
    
    mol_id_present_config = {}
    for item in tqdm(data):

        item['dup_identifier'] = '_'.join(
          [
            item['charge_spin'],
            item['sp_config_type'],
            item['optimized_parent_charge_spin'],
            item['solvent']
          ]
        )
        
        try:
            mol_id_present_config[item['mol_id']].append(item['dup_identifier'])
        except KeyError:
            mol_id_present_config[item['mol_id']] = [item['dup_identifier']]
    
    # get unique set of configs for each key in mol_id_present_config to use as keys to sample from bucket_mol_id
    for mol_id in tqdm(bucket_mol_id):
        pool = list(set(mol_id_present_config[mol_id]))

        # Data missing! Exclude molecule
        if len(pool) < 6:
            continue

        for item in pool:
            options = [p for p in bucket_mol_id[mol_id] if p['dup_identifier'] == item]
            if len(options) > 0:
                # Always pick the lowest-energy point
                # Should be the best SCF solution
                filtered_data.append(min(options, key=lambda x: x['energy']))

    return filtered_data


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
        resp_partial_charges = np.array(item['resp_partial_charges'])
        geometry = np.array(item['geometry'])

        dipole_components = resp_partial_charges[:, np.newaxis] * geometry
        dipole_moment = np.sum(dipole_components, axis=0) * (1 / 0.208193)

        item['calc_resp_dipole_moment'] = dipole_moment.tolist()


def resolve_partial_spins(data: list):
    for item in tqdm(data):
        if int(item['spin']) == 1:
            charge_array_shape = np.array(item['mulliken_partial_charges']).shape
            if item['mulliken_partial_spins'] is None or None in item['mulliken_partial_spins']:
                item['mulliken_partial_spins'] = np.zeros(charge_array_shape, dtype=float).tolist()

            # if item['nbo_partial_spins'] is None or None in item['nbo_partial_spins']:
            #     if item['nbo_partial_charges'] is not None and None not in item['nbo_partial_charges']:
            #         item['nbo_partial_spins'] = np.zeros(charge_array_shape, dtype=float).tolist()


def force_magnitude_filter(cutoff: float,
                           data: list):
    """
    This method returns both data that meets the cuttoff value and data that is equal to or above the cuttoff value.
    If this is run before downsampling, it removes the entire data point trajectory.
    
    Returns: lists
    """
    good = []
    for item in tqdm(data):
        forces = item['gradient']
        problem = False
        for atom in forces:
            try:
                res = np.sqrt(sum([i**2 for i in atom]))
                if res >= cutoff:
                    problem = True
                    break
            except TypeError:
                res = np.sqrt(sum([i**2 for i in atom[0]]))
                if res >= cutoff:
                    problem = True
                    break
        if not problem:
            good.append(item)
                            
    return good


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
            graph = build_graph(item['species'], item['geometry'])
            connected = nx.is_connected(graph.graph.to_undirected())
                
            if connected:
                good.append(item)
            else:
                broken.append(item)

    return good, broken

            
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
        item['energy'] *= 27.2114

        forces = item['gradient']
        atom_arr = []
        for atom in forces:
            comp_arr = []
            for component in atom:
                new_component = component * 51.42208619083232
                comp_arr.append(new_component)
            atom_arr.append(comp_arr)
        item['gradient'] = atom_arr

        forces = item["precise_gradient"]
        atom_arr = []
        for atom in forces:
            comp_arr = []
            for component in atom:
                new_component = component * 51.42208619083232
                comp_arr.append(new_component)
            atom_arr.append(comp_arr)
        item['precise_gradient'] = atom_arr


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

    atoms = ase.atoms.Atoms(
        symbols=data['species'],
        positions=data['geometry']
    )

    length = len(data["species"])

    atoms.arrays['mulliken_partial_charges'] = np.array(data['mulliken_partial_charges'])
    atoms.arrays['mulliken_partial_spins'] = np.array(data['mulliken_partial_spins'])
    atoms.arrays['resp_partial_charges'] = np.array(data['resp_partial_charges'])
    
    # if (
    #     data["nbo_partial_charges"] is not None
    #     and None not in data["nbo_partial_charges"]
    #     and len(data["nbo_partial_charges"]) == length
    # ):
    #     atoms.arrays['nbo_partial_charges'] = np.array(data['nbo_partial_charges'])

    # if (
    #     data["nbo_partial_spins"] is not None
    #     and None not in data["nbo_partial_spins"]
    #     and len(data["nbo_partial_spins"]) == length
    # ):
    #     atoms.arrays['nbo_partial_spins'] = np.array(data['nbo_partial_spins'])

    atoms.info['dipole_moment'] = np.array(data['dipole_moment'])
    atoms.info['resp_dipole_moment'] = np.array(data['resp_dipole_moment'])
    # atoms.info['calc_resp_dipole_moment'] = np.array(data['calc_resp_dipole_moment'])
    atoms.info['weight'] = data['weight']
        
    if energy is not None:
        atoms.info['REF_energy'] = data[energy]
    if forces is not None:
        atoms.arrays['REF_forces'] = np.array(data[forces])
    if charge is not None:
            atoms.info['charge'] = data[charge]
    if spin is not None:
        atoms.info['spin'] = data[spin]
    atoms.info['mol_id'] = data['mol_id']
    atoms.info['sp_config_type'] = data['sp_config_type']

    return atoms


def build_atoms_iterator(
    data: list,
    energy: str = "energy",
    forces: str = "precise_gradient",
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
        data_set.append(atoms)
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
    ase.io.write(train_file, train_data, format="extxyz")
     
    val_file = os.path.join(path,file_name+'_val.xyz')
    ase.io.write(val_file, val_data, format="extxyz")
    
    test_file = os.path.join(path,file_name+'_test.xyz')
    ase.io.write(test_file, test_data, format="extxyz")


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


if __name__ == "__main__"

    # Charge/spin subsets
    smd_train_cs_dict = {}
    for item in tqdm(smd_build_full['train']):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            smd_train_cs_dict[key].append(item)
        except KeyError:
            smd_train_cs_dict[key] = [item]

    smd_val_cs_dict = {}
    for item in tqdm(smd_build_full['val']):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            smd_val_cs_dict[key].append(item)
        except KeyError:
            smd_val_cs_dict[key] = [item]

    smd_test_cs_dict = {}
    for item in tqdm(smd_build_full['test']):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            smd_test_cs_dict[key].append(item)
        except KeyError:
            smd_test_cs_dict[key] = [item]

    smd_ood_cs_dict = {}
    for item in tqdm(smd_ood_full):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            smd_ood_cs_dict[key].append(item)
        except KeyError:
            smd_ood_cs_dict[key] = [item]

    # Split by charge/spin pair
    # Use this for relative energies
    smd_full_chargespin_path = os.path.join(smd_full_path, "by_charge_spin")
    if not os.path.exists(smd_full_chargespin_path):
        os.mkdir(smd_full_chargespin_path)

    for key in smd_test_cs_dict:
        file = os.path.join(smd_full_chargespin_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_train_'+key+'.xyz')
        ase.io.write(file, smd_train_cs_dict[key], format="extxyz")
        
        file = os.path.join(smd_full_chargespin_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_val_'+key+'.xyz')
        ase.io.write(file, smd_val_cs_dict[key], format="extxyz")
        
        file = os.path.join(smd_full_chargespin_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_test_'+key+'.xyz')
        ase.io.write(file, smd_test_cs_dict[key], format="extxyz")

        if key in smd_ood_cs_dict:
            file = os.path.join(smd_full_chargespin_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_ood_'+key+'.xyz')
            ase.io.write(file, smd_ood_cs_dict[key], format="extxyz")

    # Singlet
    smd_full_singlet_path = os.path.join(smd_full_path, "singlet")
    if not os.path.exists(smd_full_singlet_path):
        os.mkdir(smd_full_singlet_path)

    smd_singlet_train = []
    smd_singlet_val = []
    smd_singlet_test = []
    smd_singlet_ood = []

    for item in tqdm(smd_build_full['train']):
        if item.info['spin'] == 1:
            smd_singlet_train.append(item)

    for item in tqdm(smd_build_full['val']):
        if item.info['spin'] == 1:
            smd_singlet_val.append(item)

    for item in tqdm(smd_build_full['test']):
        if item.info['spin'] == 1:
            smd_singlet_test.append(item)

    for item in tqdm(smd_ood_full):
        if item.info['spin'] == 1:
            smd_singlet_ood.append(item)

    file = os.path.join(smd_full_singlet_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_singlet_train.xyz')
    ase.io.write(file, smd_singlet_train, format="extxyz")
    
    file = os.path.join(smd_full_singlet_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_singlet_val.xyz')
    ase.io.write(file, smd_singlet_val, format="extxyz")
    
    file = os.path.join(smd_full_singlet_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_singlet_test.xyz')
    ase.io.write(file, smd_singlet_test, format="extxyz")

    file = os.path.join(smd_full_singlet_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_singlet_ood.xyz')
    ase.io.write(file, smd_singlet_ood, format="extxyz")

    # Doublet
    smd_full_doublet_path = os.path.join(smd_full_path, "doublet")
    if not os.path.exists(smd_full_doublet_path):
        os.mkdir(smd_full_doublet_path)

    smd_doublet_train = []
    smd_doublet_val = []
    smd_doublet_test = []
    smd_doublet_ood = []

    for item in tqdm(smd_build_full['train']):
        if item.info['spin'] == 2:
            smd_doublet_train.append(item)

    for item in tqdm(smd_build_full['val']):
        if item.info['spin'] == 2:
            smd_doublet_val.append(item)

    for item in tqdm(smd_build_full['test']):
        if item.info['spin'] == 2:
            smd_doublet_test.append(item)

    for item in tqdm(smd_ood_full):
        if item.info['spin'] == 2:
            smd_doublet_ood.append(item)

    file = os.path.join(smd_full_doublet_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_doublet_train.xyz')
    ase.io.write(file, smd_doublet_train, format="extxyz")
    
    file = os.path.join(smd_full_doublet_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_doublet_val.xyz')
    ase.io.write(file, smd_doublet_val, format="extxyz")
    
    file = os.path.join(smd_full_doublet_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_doublet_test.xyz')
    ase.io.write(file, smd_doublet_test, format="extxyz")

    file = os.path.join(smd_full_doublet_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_doublet_ood.xyz')
    ase.io.write(file, smd_doublet_ood, format="extxyz")


    # Neutral
    smd_full_neutral_path = os.path.join(smd_full_path, "neutral")
    if not os.path.exists(smd_full_neutral_path):
        os.mkdir(smd_full_neutral_path)

    smd_neutral_train = []
    smd_neutral_val = []
    smd_neutral_test = []
    smd_neutral_ood = []

    for item in tqdm(smd_build_full['train']):
        if item.info['charge'] == 0:
            smd_neutral_train.append(item)

    for item in tqdm(smd_build_full['val']):
        if item.info['charge'] == 0:
            smd_neutral_val.append(item)

    for item in tqdm(smd_build_full['test']):
        if item.info['charge'] == 0:
            smd_neutral_test.append(item)

    for item in tqdm(smd_ood_full):
        if item.info['charge'] == 0:
            smd_neutral_ood.append(item)

    file = os.path.join(smd_full_neutral_path,'radqm9_65_10_25_sp_smd_full_data_20240807_neutral_train.xyz')
    ase.io.write(file, smd_neutral_train, format="extxyz")
    
    file = os.path.join(smd_full_neutral_path,'radqm9_65_10_25_sp_smd_full_data_20240807_neutral_val.xyz')
    ase.io.write(file, smd_neutral_val, format="extxyz")
    
    file = os.path.join(smd_full_neutral_path,'radqm9_65_10_25_sp_smd_full_data_20240807_neutral_test.xyz')
    ase.io.write(file, smd_neutral_test, format="extxyz")

    file = os.path.join(smd_full_neutral_path,'radqm9_65_10_25_sp_smd_full_data_20240807_neutral_ood.xyz')
    ase.io.write(file, smd_neutral_ood, format="extxyz")

    fractions = [.01, .05, .1, .25, .5, .75]

    wtd_full = weight_to_data_ase(smd_build_full["train"])
    cd_full = chunk_data(wtd_full, fractions)
    
    wtd_cs = dict()
    cd_cs = dict()
    for key in smd_train_cs_dict:
        wtd_cs[key] = weight_to_data_ase(smd_train_cs_dict[key])
        cd_cs[key] = chunk_data(wtd_cs[key], fractions)

    wtd_singlet = weight_to_data_ase(smd_singlet_train)
    cd_singlet = chunk_data(wtd_singlet, fractions)

    wtd_doublet = weight_to_data_ase(smd_doublet_train)
    cd_doublet = chunk_data(wtd_doublet, fractions)

    wtd_neutral = weight_to_data_ase(smd_neutral_train)
    cd_neutral = chunk_data(wtd_neutral, fractions)

    for ii, frac in enumerate(fractions):
        chunk_file = os.path.join(smd_full_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_full[ii], format="extxyz")
        
        for key in cd_cs:
            chunk_file = os.path.join(smd_full_chargespin_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_train_subset_' + key + f'_{frac}.xyz')
            ase.io.write(chunk_file, cd_cs[key][ii], format="extxyz")

        chunk_file = os.path.join(smd_full_singlet_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_singlet_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_singlet[ii],format="extxyz")

        chunk_file = os.path.join(smd_full_doublet_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_doublet_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_doublet[ii],format="extxyz")
        
        chunk_file = os.path.join(smd_full_neutral_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_neutral_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_neutral[ii],format="extxyz")
