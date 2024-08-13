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
          item['charge_spin'],
          item['sp_config_type'],
          item['optimized_parent_charge_spin'],
          item['solvent']  
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

        dipole_components = resp_partial_charges[:, np.newaxis] * geometries
        dipole_moment = np.sum(dipole_components, axis=0) * (1 / 0.208193)

        item['calc_resp_dipole_moment'] = dipole_moment.tolist()


def resolve_mulliken_partial_spins(data: list):
    for item in tqdm(data):
        if item['charge_spin']=='0_1':
            if item['mulliken_partial_spins'] is None or None in item['mulliken_partial_spins']:
                charge_array = np.array(item['mulliken_partial_charges'])
                item['mulliken_partial_spins'] = np.zeros(charge_array.shape, dtype=float).tolist()


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
        for atom in forces:
            try:
                res = np.sqrt(sum([i**2 for i in atom]))
                if res >= cutoff:
                    bad.append(item)
                    next_item = True
                    break
            except TypeError:
                res = np.sqrt(sum([i**2 for i in atom[0]]))
                if res >= cutoff:
                    bad.append(item)
                    next_item = True
                    break
        if not next_item:
            good.append(item)
                            
    return good


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


if __name__ == "__main__":

    base_path = ""

    vacuum_data_path = os.path.join(base_path, "vacuum")
    vacuum_minimal_path = os.path.join(vacuum_data_path, "minimal")
    vacuum_full_path = os.path.join(vacuum_data_path, "full")

    smd_data_path = os.path.join(base_path, "smd")
    smd_minimal_path = os.path.join(smd_data_path, "minimal")
    smd_full_path = os.path.join(smd_data_path, "full")

    for path in [
        base_path,
        vacuum_data_path,
        smd_data_path,
        vacuum_minimal_path,
        vacuum_full_path,
        smd_minimal_path,
        smd_full_path
    ]:
        if not os.path.exists(path):
            os.mkdir(path)

    elements_dict = read_elements('/global/home/users/ewcspottesmith/software/radqm9_pipeline/src/radqm9_pipeline/elements/elements.pkl')

    # Trajectory information
    force_store = MongoStore(database="thermo_chem_storage",
                            collection_name="radqm9_force",
                            username="thermo_chem_storage_ro",
                            password="",
                            host="mongodb07.nersc.gov",
                            port=27017,
                            key="molecule_id")
    force_store.connect()

    data = []
    for entry in tqdm(
        force_store.query(
            {
                "precise_forces": {  # Tossing out 7513 datapoints; we can live with that
                    "$ne": None
                }
            },
            {
                "molecule_id": 1,
                "species": 1,
                "charge": 1,
                "spin_multiplicity": 1,
                "coordinates": 1,
                "energy": 1,
                "forces": 1,
                "mulliken_partial_charges": 1,
                "mulliken_partial_spins": 1,
                "resp_partial_charges": 1,
                "dipole_moment": 1,
                "resp_dipole_moment": 1,
                "nbo_partial_charges": 1,  # OPTIONAL
                "nbo_partial_spins": 1,  # OPTIONAL
                "precise_forces": 1,
                "solvent": 1,
            }
        )
    ):
        item = {}
        item['mol_id'] = entry['molecule_id']
        item['species'] = entry['species']
        item['charge'] = entry['charge'] 
        item['spin'] = entry['spin_multiplicity']
        item['geometry'] = entry['coordinates']
        item['energy'] = entry['energy']
        item['gradient'] = entry['forces']
        item['precise_gradient'] = entry['precise_forces']
        item['mulliken_partial_charges'] = entry['mulliken_partial_charges']
        item['mulliken_partial_spins'] = entry['mulliken_partial_spins']
        item['resp_partial_charges'] = entry['resp_partial_charges']
        item['dipole_moment'] = entry['dipole_moment']
        item['resp_dipole_moment'] = entry['resp_dipole_moment']
        item['nbo_partial_charges'] = entry['nbo_partial_charges']
        item['nbo_partial_spins'] = entry['nbo_partial_spins']
        
        if entry['solvent'] == "NONE":
            item['solvent'] = 'vacuum'
        else:
            item['solvent'] = "SMD"

        item['charge_spin'] = str(int(item['charge'])) + '_' + str(int(item['spin']))

        molid_contents = item['mol_id'].split('-')
        parent_charge = molid_contents[2].replace("m", "-")
        parent_spin = molid_contents[3]
        item['optimized_parent_charge_spin'] = parent_charge + '_' + parent_spin
        if item['charge_spin'] == item['optimized_parent_charge_spin']:
            item['sp_config_type'] = 'optimized'
        else:
            item['sp_config_type'] = 'vertical'

        data.append(item)

    generate_resp_dipole(data)

    resolve_mulliken_partial_spins(data)

    dumpfn(data, os.path.join(base_path, "raw_sp_data.json"))

    data = filter_duplicate_and_missing_data(data)

    data = force_magnitude_filter(cutoff=10.0, data=data)

    convert_energy_forces(data)

    molecule_weight(data, elements_dict)

    vacuum_data = []
    smd_data = []
    for item in data:
        solv = item['solvent']
        
        if solv == 'vacuum':
            vacuum_data.append(item)
        
        elif solv == 'SMD':
            smd_data.append(item)

    dumpfn(vacuum_data, os.path.join(vacuum_data_path, "filtered_vacuum_sp_data.json"))
    dumpfn(smd_data, os.path.join(smd_data_path, "filtered_smd_sp_data.json"))

    vacuum_data, vacuum_ood = filter_broken_graphs(vacuum_data)
    smd_data, smd_ood = filter_broken_graphs(smd_data)

    # TODO: you are here

    # Vacuum data

    wtd = weight_to_data(vacuum_data)
    sld = length_dict(wtd)

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

    train_subset={key: sld[key] for key in train_mass if key in sld}
    test_subset={key: sld[key] for key in test_mass if key in sld}
    val_subset={key: sld[key] for key in val_mass if key in sld}

    train_temp=[[x]*train_subset[x] for x in train_subset]
    test_temp=[[x]*test_subset[x] for x in test_subset]
    val_temp=[[x]*val_subset[x] for x in val_subset]

    train_subset_merged = list(chain.from_iterable(train_temp))
    test_subset_merged = list(chain.from_iterable(test_temp))
    val_subset_merged = list(chain.from_iterable(val_temp))

    distribution = {
        "train": train_subset_merged,
        "val": val_subset_merged,
        "test": test_subset_merged
    }

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

    # Split by charge/spin pair
    # Use this for relative energies
    minimal_chargespin_path = os.path.join(minimal_data_path, "by_charge_spin")
    if not os.path.exists(minimal_chargespin_path):
        os.mkdir(minimal_chargespin_path)

    for key in test_cs_dict:
        file = os.path.join(minimal_chargespin_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_train'+key+'.xyz')
        ase.io.write(file, train_cs_dict[key], format="extxyz")
        
        file = os.path.join(minimal_chargespin_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_val'+key+'.xyz')
        ase.io.write(file, val_cs_dict[key],format="extxyz")
        
        file = os.path.join(minimal_chargespin_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_test'+key+'.xyz')
        ase.io.write(file, test_cs_dict[key],format="extxyz")

    # Doublet
    minimal_doublet_path = os.path.join(minimal_data_path, "doublet")
    if not os.path.exists(minimal_doublet_path):
        os.mkdir(minimal_doublet_path)

    doublet_train = []
    doublet_val = []
    doublet_test = []

    for item in tqdm(build_minimal['train']):
        if item.info['spin'] == 2:
            doublet_train.append(item)

    for item in tqdm(build_minimal['val']):
        if item.info['spin'] == 2:
            doublet_val.append(item)

    for item in tqdm(build_minimal['test']):
        if item.info['spin'] == 2:
            doublet_test.append(item)

    file = os.path.join(minimal_doublet_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_doublet_train.xyz')
    ase.io.write(file, doublet_train, format="extxyz")
    
    file = os.path.join(minimal_doublet_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_doublet_val.xyz')
    ase.io.write(file, doublet_val,format="extxyz")
    
    file = os.path.join(minimal_doublet_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_doublet_test.xyz')
    ase.io.write(file, doublet_test,format="extxyz")

    # Neutral
    minimal_neutral_path = os.path.join(minimal_data_path, "neutral")
    if not os.path.exists(minimal_neutral_path):
        os.mkdir(minimal_neutral_path)

    neutral_train = []
    neutral_val = []
    neutral_test = []

    for item in tqdm(build_minimal['train']):
        if item.info['charge'] == 0:
            neutral_train.append(item)

    for item in tqdm(build_minimal['val']):
        if item.info['charge'] == 0:
            neutral_val.append(item)

    for item in tqdm(build_minimal['test']):
        if item.info['charge'] == 0:
            neutral_test.append(item)

    file = os.path.join(minimal_neutral_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_neutral_train.xyz')
    ase.io.write(file, neutral_train, format="extxyz")
    
    file = os.path.join(minimal_neutral_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_neutral_val.xyz')
    ase.io.write(file, neutral_val,format="extxyz")
    
    file = os.path.join(minimal_neutral_path,'radqm9_65_10_25_trajectory_minimal_data_20240807_neutral_test.xyz')
    ase.io.write(file, neutral_test,format="extxyz")

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
        chunk_file = os.path.join(minimal_data_path, 'radqm9_65_10_25_trajectory_minimal_data_20240807_neutral_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_minimal[ii],format="extxyz")
        
        for key in cd_cs:
            chunk_file = os.path.join(minimal_chargespin_path, 'radqm9_65_10_25_trajectory_minimal_data_20240807_train_subset_' + key + f'_{frac}.xyz')
            ase.io.write(chunk_file, cd_cs[key][ii], format="extxyz")
            
        chunk_file = os.path.join(minimal_doublet_path, 'radqm9_65_10_25_trajectory_minimal_data_20240807_doublet_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_doublet[ii],format="extxyz")
        
        chunk_file = os.path.join(minimal_neutral_path, 'radqm9_65_10_25_trajectory_minimal_data_20240807_neutral_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_neutral[ii],format="extxyz")

    # Full build
    build_full = dict()
    for split in data:
        build_full[split] = build_atoms_iterator(data[split], energy="energies")
        
    create_dataset(build_full, 'radqm9_65_10_25_trajectory_full_data_20240807', full_data_path)

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

    # Split by charge/spin pair
    # Use this for relative energies
    full_chargespin_path = os.path.join(full_data_path, "by_charge_spin")
    if not os.path.exists(full_chargespin_path):
        os.mkdir(full_chargespin_path)

    for key in test_cs_dict:
        file = os.path.join(full_chargespin_path,'radqm9_65_10_25_trajectory_full_data_20240807_train'+key+'.xyz')
        ase.io.write(file, train_cs_dict[key], format="extxyz")
        
        file = os.path.join(minimal_chargespin_path,'radqm9_65_10_25_trajectory_full_data_20240807_val'+key+'.xyz')
        ase.io.write(file, val_cs_dict[key],format="extxyz")
        
        file = os.path.join(minimal_chargespin_path,'radqm9_65_10_25_trajectory_full_data_20240807_test'+key+'.xyz')
        ase.io.write(file, test_cs_dict[key],format="extxyz")

    # Doublet
    full_doublet_path = os.path.join(full_data_path, "doublet")
    if not os.path.exists(full_doublet_path):
        os.mkdir(full_doublet_path)

    doublet_train = []
    doublet_val = []
    doublet_test = []

    for item in tqdm(build_minimal['train']):
        if item.info['spin'] == 2:
            doublet_train.append(item)

    for item in tqdm(build_minimal['val']):
        if item.info['spin'] == 2:
            doublet_val.append(item)

    for item in tqdm(build_minimal['test']):
        if item.info['spin'] == 2:
            doublet_test.append(item)

    file = os.path.join(full_doublet_path,'radqm9_65_10_25_trajectory_full_data_20240807_doublet_train.xyz')
    ase.io.write(file, doublet_train, format="extxyz")
    
    file = os.path.join(full_doublet_path,'radqm9_65_10_25_trajectory_full_data_20240807_doublet_val.xyz')
    ase.io.write(file, doublet_val,format="extxyz")
    
    file = os.path.join(full_doublet_path,'radqm9_65_10_25_trajectory_full_data_20240807_doublet_test.xyz')
    ase.io.write(file, doublet_test,format="extxyz")

    # Neutral
    full_neutral_path = os.path.join(full_data_path, "neutral")
    if not os.path.exists(full_neutral_path):
        os.mkdir(full_neutral_path)

    neutral_train = []
    neutral_val = []
    neutral_test = []

    for item in tqdm(build_minimal['train']):
        if item.info['charge'] == 0:
            neutral_train.append(item)

    for item in tqdm(build_minimal['val']):
        if item.info['charge'] == 0:
            neutral_val.append(item)

    for item in tqdm(build_minimal['test']):
        if item.info['charge'] == 0:
            neutral_test.append(item)

    file = os.path.join(full_neutral_path,'radqm9_65_10_25_trajectory_full_data_20240807_neutral_train.xyz')
    ase.io.write(file, neutral_train, format="extxyz")
    
    file = os.path.join(full_neutral_path,'radqm9_65_10_25_trajectory_full_data_20240807_neutral_val.xyz')
    ase.io.write(file, neutral_val,format="extxyz")
    
    file = os.path.join(full_neutral_path,'radqm9_65_10_25_trajectory_full_data_20240807_neutral_test.xyz')
    ase.io.write(file, neutral_test,format="extxyz")

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
        chunk_file = os.path.join(full_data_path, 'radqm9_65_10_25_trajectory_full_data_20240807_neutral_train_subset_' + f'{frac}.xyz')
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