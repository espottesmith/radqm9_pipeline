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

        dipole_components = resp_partial_charges[:, np.newaxis] * geometry
        dipole_moment = np.sum(dipole_components, axis=0) * (1 / 0.208193)

        item['calc_resp_dipole_moment'] = dipole_moment.tolist()


def resolve_partial_spins(data: list):
    for item in tqdm(data):
        if item['charge_spin']=='0_1':
            if item['mulliken_partial_spins'] is None or None in item['mulliken_partial_spins']:
                charge_array = np.array(item['mulliken_partial_charges'])
                item['mulliken_partial_spins'] = np.zeros(charge_array.shape, dtype=float).tolist()

            if item['nbo_partial_spins'] is None or None in item['nbo_partial_spins']:
                if item['nbo_partial_charges'] is not None and None not in item['nbo_partial_charges']:
                    charge_array = np.array(item['nbo_partial_charges'])
                    item['nbo_partial_spins'] = np.zeros(charge_array.shape, dtype=float).tolist()


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

    atoms.arrays['mulliken_partial_charges'] = np.array(data['mulliken_partial_charges'])
    atoms.arrays['mulliken_partial_spins'] = np.array(data['mulliken_partial_spins'])
    atoms.arrays['resp_partial_charges'] = np.array(data['resp_partial_charges'])
    atoms.arrays['nbo_partial_charges'] = np.array(data['nbo_partial_charges'])
    atoms.arrays['nbo_partial_spins'] = np.array(data['nbo_partial_spins'])

    atoms.info['dipole_moment'] = np.array(data['dipole_moment'])
    atoms.info['resp_dipole_moment'] = np.array(data['resp_dipole_moment'])
    atoms.info['calc_resp_dipole_moment']=np.array(data['calc_resp_dipole_moment'])
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

    return atoms


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

    atoms = ase.atoms.Atoms(
        symbols=data['species'],
        positions=data['geometry']
    )
        
    atoms.info['weight'] = data['weight']

    if energy is not None:
        atoms.info['energy'] = data[energy]
    if forces is not None:
        atoms.arrays['forces'] = np.array(data[forces])
    if charge is not None:
        atoms.info['charge'] = data[charge]
    if spin is not None:
        atoms.info['spin'] = data[spin]
    atoms.info['mol_id'] = data['mol_id']

    atoms.info['sp_config_type'] = atoms.info['sp_config_type']

    return atoms


def build_atoms_iterator(
    data: list,
    energy: str = "energy",
    forces: str = "gradient",
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
    forces: str = "gradient",
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

    resolve_partial_spins(data)

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

    # Vacuum data

    wtd = weight_to_data(vacuum_data)
    sld = length_dict(wtd)

    vac_train_mass = ['152.037'] # EVAN WILL NEED TO ADJUST THE MASSES OF INITIAL POINTS FOR NEW DATA
    vac_test_mass = ['144.092']
    vac_val_mass = ['143.108']

    vac_train = sld['152.037'] # trackers for dataset sizes
    vac_test = sld['144.092']
    vac_val = sld['143.108']

    sld.pop('152.037')
    sld.pop('144.092')
    sld.pop('143.108')

    # Sort the data 
    # data is a dict: mass-># of trajs
    for mass in sld:
        temp_total = vac_train + vac_val + vac_test
        train_ratio = .65 - (vac_train / temp_total)
        test_ratio = .25 - (vac_test / temp_total)
        val_ratio = .1 - (vac_val / temp_total)
        
        if train_ratio > val_ratio and train_ratio > test_ratio:
            vac_train_mass.append(mass)
            vac_train += sld[mass]
        elif val_ratio > train_ratio and val_ratio > test_ratio:
            vac_val_mass.append(mass)
            vac_val += sld[mass]
        else:
            vac_test_mass.append(mass)
            vac_test += sld[mass]

    sld = length_dict(wtd) # you need to call this again yes

    vac_switch = [
        "117.039",
        "116.204",
        "116.160",
        "115.096",
        "115.095",
        "112.054",
        "112.05",
        "111.148",
        "102.089",
        "101.065",
        "101.061",
        "100.205",
        "99.053",
        "99.049",
        "98.189",
        "95.023",
        "94.117",
        "85.106",
        "84.078",
        "83.046"
    ]

    for mass in vac_switch:
        vac_val_mass.append(mass)
        vac_val += sorted_length_dict[mass]
        
        vac_test_mass.remove(mass)
        vac_test -= sorted_length_dict[mass]

    vac_train_data = [wtd[x] for x in vac_train_mass]
    vac_train_data = list(chain.from_iterable(vac_train_data))

    vac_val_data = [wtd[x] for x in vac_val_mass]
    vac_val_data = list(chain.from_iterable(vac_val_data))

    vac_test_data = [wtd[x] for x in vac_test_mass]
    vac_test_data = list(chain.from_iterable(vac_test_data))

    vac_data = {
        'train':vac_train_data,
        'val': vac_val_data,
        'test': vac_test_data
    }

    # Minimal build
    vac_build_minimal = dict()
    for split in data:
        vac_build_minimal[split] = build_minimal_atoms_iterator(vac_data[split], forces="precise_gradient")
        
    create_dataset(vac_build_minimal, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807', vacuum_minimal_path)

    vac_ood_minimal = build_minimal_atoms_iterator(vacuum_ood, forces="precise_gradient")
    file = os.path.join(vacuum_minimal_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_ood.xyz')
    ase.io.write(file, vac_ood_minimal, format="extxyz")

    # Charge/spin subsets
    vac_train_cs_dict = {}
    for item in tqdm(vac_build_minimal['train']):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            vac_train_cs_dict[key].append(item)
        except KeyError:
            vac_train_cs_dict[key] = [item]

    vac_val_cs_dict = {}
    for item in tqdm(vac_build_minimal['val']):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            vac_val_cs_dict[key].append(item)
        except KeyError:
            vac_val_cs_dict[key] = [item]

    vac_test_cs_dict = {}
    for item in tqdm(vac_build_minimal['test']):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            vac_test_cs_dict[key].append(item)
        except KeyError:
            vac_test_cs_dict[key] = [item]

    vac_ood_cs_dict = {}
    for item in tqdm(vac_ood_minimal):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            vac_ood_cs_dict[key].append(item)
        except KeyError:
            vac_ood_cs_dict[key] = [item]

    # Split by charge/spin pair
    # Use this for relative energies
    vacuum_minimal_chargespin_path = os.path.join(vacuum_minimal_path, "by_charge_spin")
    if not os.path.exists(vacuum_minimal_chargespin_path):
        os.mkdir(vacuum_minimal_chargespin_path)

    for key in vac_test_cs_dict:
        file = os.path.join(vacuum_minimal_chargespin_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_train_'+key+'.xyz')
        ase.io.write(file, vac_train_cs_dict[key], format="extxyz")
        
        file = os.path.join(vacuum_minimal_chargespin_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_val_'+key+'.xyz')
        ase.io.write(file, vac_val_cs_dict[key], format="extxyz")
        
        file = os.path.join(vacuum_minimal_chargespin_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_test_'+key+'.xyz')
        ase.io.write(file, vac_test_cs_dict[key], format="extxyz")

        if key in vac_ood_cs_dict:
            file = os.path.join(vacuum_minimal_chargespin_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_ood_'+key+'.xyz')
            ase.io.write(file, vac_ood_cs_dict[key], format="extxyz")

    # Singlet
    vacuum_minimal_singlet_path = os.path.join(vacuum_minimal_path, "singlet")
    if not os.path.exists(vacuum_minimal_singlet_path):
        os.mkdir(vacuum_minimal_singlet_path)

    vac_singlet_train = []
    vac_singlet_val = []
    vac_singlet_test = []
    vac_singlet_ood = []

    for item in tqdm(vac_build_minimal['train']):
        if item.info['spin'] == 1:
            vac_singlet_train.append(item)

    for item in tqdm(vac_build_minimal['val']):
        if item.info['spin'] == 1:
            vac_singlet_val.append(item)

    for item in tqdm(vac_build_minimal['test']):
        if item.info['spin'] == 1:
            vac_singlet_test.append(item)

    for item in tqdm(vac_ood_minimal):
        if item.info['spin'] == 1:
            vac_singlet_ood.append(item)

    file = os.path.join(vacuum_minimal_singlet_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_singlet_train.xyz')
    ase.io.write(file, vac_singlet_train, format="extxyz")
    
    file = os.path.join(vacuum_minimal_singlet_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_singlet_val.xyz')
    ase.io.write(file, vac_singlet_val, format="extxyz")
    
    file = os.path.join(vacuum_minimal_singlet_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_singlet_test.xyz')
    ase.io.write(file, vac_singlet_test, format="extxyz")

    file = os.path.join(vacuum_minimal_singlet_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_singlet_ood.xyz')
    ase.io.write(file, vac_singlet_ood, format="extxyz")

    # Doublet
    vacuum_minimal_doublet_path = os.path.join(vacuum_minimal_path, "doublet")
    if not os.path.exists(vacuum_minimal_doublet_path):
        os.mkdir(vacuum_minimal_doublet_path)

    vac_doublet_train = []
    vac_doublet_val = []
    vac_doublet_test = []
    vac_doublet_ood = []

    for item in tqdm(vac_build_minimal['train']):
        if item.info['spin'] == 2:
            vac_doublet_train.append(item)

    for item in tqdm(vac_build_minimal['val']):
        if item.info['spin'] == 2:
            vac_doublet_val.append(item)

    for item in tqdm(vac_build_minimal['test']):
        if item.info['spin'] == 2:
            vac_doublet_test.append(item)

    for item in tqdm(vac_ood_minimal):
        if item.info['spin'] == 2:
            vac_doublet_ood.append(item)

    file = os.path.join(vacuum_minimal_doublet_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_doublet_train.xyz')
    ase.io.write(file, vac_doublet_train, format="extxyz")
    
    file = os.path.join(vacuum_minimal_doublet_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_doublet_val.xyz')
    ase.io.write(file, vac_doublet_val, format="extxyz")
    
    file = os.path.join(vacuum_minimal_doublet_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_doublet_test.xyz')
    ase.io.write(file, vac_doublet_test, format="extxyz")

    file = os.path.join(vacuum_minimal_doublet_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_doublet_ood.xyz')
    ase.io.write(file, vac_doublet_ood, format="extxyz")


    # Neutral
    vacuum_minimal_neutral_path = os.path.join(vacuum_minimal_path, "neutral")
    if not os.path.exists(vacuum_minimal_neutral_path):
        os.mkdir(vacuum_minimal_neutral_path)

    vac_neutral_train = []
    vac_neutral_val = []
    vac_neutral_test = []
    vac_neutral_ood = []

    for item in tqdm(vac_build_minimal['train']):
        if item.info['charge'] == 0:
            vac_neutral_train.append(item)

    for item in tqdm(vac_build_minimal['val']):
        if item.info['charge'] == 0:
            vac_neutral_val.append(item)

    for item in tqdm(vac_build_minimal['test']):
        if item.info['charge'] == 0:
            vac_neutral_test.append(item)

    for item in tqdm(vac_ood_minimal):
        if item.info['charge'] == 0:
            vac_neutral_ood.append(item)

    file = os.path.join(vacuum_minimal_neutral_path,'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_neutral_train.xyz')
    ase.io.write(file, vac_neutral_train, format="extxyz")
    
    file = os.path.join(vacuum_minimal_neutral_path,'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_neutral_val.xyz')
    ase.io.write(file, vac_neutral_val, format="extxyz")
    
    file = os.path.join(vacuum_minimal_neutral_path,'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_neutral_test.xyz')
    ase.io.write(file, vac_neutral_test, format="extxyz")

    file = os.path.join(vacuum_minimal_neutral_path,'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_neutral_ood.xyz')
    ase.io.write(file, vac_neutral_ood, format="extxyz")

    fractions = [.01, .05, .1, .25, .5, .75]

    wtd_minimal = weight_to_data_ase(vac_build_minimal["train"])
    cd_minimal = chunk_data(wtd_minimal, fractions)
    
    wtd_cs = dict()
    cd_cs = dict()
    for key in vac_train_cs_dict:
        wtd_cs[key] = weight_to_data_ase(vac_train_cs_dict[key])
        cd_cs[key] = chunk_data(wtd_cs[key], fractions)

    wtd_singlet = weight_to_data_ase(vac_singlet_train)
    cd_singlet = chunk_data(wtd_singlet, fractions)

    wtd_doublet = weight_to_data_ase(vac_doublet_train)
    cd_doublet = chunk_data(wtd_doublet, fractions)

    wtd_neutral = weight_to_data_ase(vac_neutral_train)
    cd_neutral = chunk_data(wtd_neutral, fractions)

    for ii, frac in enumerate(fractions):
        chunk_file = os.path.join(vacuum_minimal_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_minimal[ii], format="extxyz")
        
        for key in cd_cs:
            chunk_file = os.path.join(vacuum_minimal_chargespin_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_train_subset_' + key + f'_{frac}.xyz')
            ase.io.write(chunk_file, cd_cs[key][ii], format="extxyz")

        chunk_file = os.path.join(vacuum_minimal_singlet_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_singlet_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_singlet[ii],format="extxyz")

        chunk_file = os.path.join(vacuum_minimal_doublet_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_doublet_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_doublet[ii],format="extxyz")
        
        chunk_file = os.path.join(vacuum_minimal_neutral_path, 'radqm9_65_10_25_sp_vacuum_minimal_data_20240807_neutral_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_neutral[ii],format="extxyz")

    # Cleanup for memory
    del vac_build_minimal
    del vac_ood_minimal
    del vac_train_cs_dict
    del vac_val_cs_dict
    del vac_test_cs_dict
    del vac_ood_cs_dict
    del vac_singlet_train
    del vac_singlet_val
    del vac_singlet_test
    del vac_singlet_ood
    del vac_doublet_train
    del vac_doublet_val
    del vac_doublet_test
    del vac_doublet_ood
    del vac_neutral_train
    del vac_neutral_val
    del vac_neutral_test
    del vac_neutral_ood
    del wtd_minimal
    del wtd_cs
    del wtd_doublet
    del wtd_neutral

    # Full build
    vac_build_full = dict()
    for split in data:
        vac_build_full[split] = build_atoms_iterator(vacuum_data[split], forces="precise_gradient")
        
    create_dataset(build_full, 'radqm9_65_10_25_sp_vacuum_full_data_20240807', vacuum_full_path)

    vac_ood_full = build_atoms_iterator(vacuum_ood, forces="precise_gradient")
    file = os.path.join(vacuum_full_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_ood.xyz')
    ase.io.write(file, vac_ood_full, format="extxyz")

    # Charge/spin subsets
    vac_train_cs_dict = {}
    for item in tqdm(vac_build_full['train']):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            vac_train_cs_dict[key].append(item)
        except KeyError:
            vac_train_cs_dict[key] = [item]

    vac_val_cs_dict = {}
    for item in tqdm(vac_build_full['val']):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            vac_val_cs_dict[key].append(item)
        except KeyError:
            vac_val_cs_dict[key] = [item]

    vac_test_cs_dict = {}
    for item in tqdm(vac_build_full['test']):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            vac_test_cs_dict[key].append(item)
        except KeyError:
            vac_test_cs_dict[key] = [item]

    vac_ood_cs_dict = {}
    for item in tqdm(vac_ood_full):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            vac_ood_cs_dict[key].append(item)
        except KeyError:
            vac_ood_cs_dict[key] = [item]

    # Split by charge/spin pair
    # Use this for relative energies
    vacuum_full_chargespin_path = os.path.join(vacuum_full_path, "by_charge_spin")
    if not os.path.exists(vacuum_full_chargespin_path):
        os.mkdir(vacuum_full_chargespin_path)

    for key in vac_test_cs_dict:
        file = os.path.join(vacuum_full_chargespin_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_train_'+key+'.xyz')
        ase.io.write(file, vac_train_cs_dict[key], format="extxyz")
        
        file = os.path.join(vacuum_full_chargespin_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_val_'+key+'.xyz')
        ase.io.write(file, vac_val_cs_dict[key], format="extxyz")
        
        file = os.path.join(vacuum_full_chargespin_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_test_'+key+'.xyz')
        ase.io.write(file, vac_test_cs_dict[key], format="extxyz")

        if key in vac_ood_cs_dict:
            file = os.path.join(vacuum_full_chargespin_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_ood_'+key+'.xyz')
            ase.io.write(file, vac_ood_cs_dict[key], format="extxyz")

    # Singlet
    vacuum_full_singlet_path = os.path.join(vacuum_full_path, "singlet")
    if not os.path.exists(vacuum_full_singlet_path):
        os.mkdir(vacuum_full_singlet_path)

    vac_singlet_train = []
    vac_singlet_val = []
    vac_singlet_test = []
    vac_singlet_ood = []

    for item in tqdm(vac_build_full['train']):
        if item.info['spin'] == 1:
            vac_singlet_train.append(item)

    for item in tqdm(vac_build_full['val']):
        if item.info['spin'] == 1:
            vac_singlet_val.append(item)

    for item in tqdm(vac_build_full['test']):
        if item.info['spin'] == 1:
            vac_singlet_test.append(item)

    for item in tqdm(vac_ood_full):
        if item.info['spin'] == 1:
            vac_singlet_ood.append(item)

    file = os.path.join(vacuum_full_singlet_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_singlet_train.xyz')
    ase.io.write(file, vac_singlet_train, format="extxyz")
    
    file = os.path.join(vacuum_full_singlet_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_singlet_val.xyz')
    ase.io.write(file, vac_singlet_val, format="extxyz")
    
    file = os.path.join(vacuum_full_singlet_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_singlet_test.xyz')
    ase.io.write(file, vac_singlet_test, format="extxyz")

    file = os.path.join(vacuum_full_singlet_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_singlet_ood.xyz')
    ase.io.write(file, vac_singlet_ood, format="extxyz")

    # Doublet
    vacuum_full_doublet_path = os.path.join(vacuum_full_path, "doublet")
    if not os.path.exists(vacuum_full_doublet_path):
        os.mkdir(vacuum_full_doublet_path)

    vac_doublet_train = []
    vac_doublet_val = []
    vac_doublet_test = []
    vac_doublet_ood = []

    for item in tqdm(vac_build_full['train']):
        if item.info['spin'] == 2:
            vac_doublet_train.append(item)

    for item in tqdm(vac_build_full['val']):
        if item.info['spin'] == 2:
            vac_doublet_val.append(item)

    for item in tqdm(vac_build_full['test']):
        if item.info['spin'] == 2:
            vac_doublet_test.append(item)

    for item in tqdm(vac_ood_full):
        if item.info['spin'] == 2:
            vac_doublet_ood.append(item)

    file = os.path.join(vacuum_full_doublet_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_doublet_train.xyz')
    ase.io.write(file, vac_doublet_train, format="extxyz")
    
    file = os.path.join(vacuum_full_doublet_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_doublet_val.xyz')
    ase.io.write(file, vac_doublet_val, format="extxyz")
    
    file = os.path.join(vacuum_full_doublet_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_doublet_test.xyz')
    ase.io.write(file, vac_doublet_test, format="extxyz")

    file = os.path.join(vacuum_full_doublet_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_doublet_ood.xyz')
    ase.io.write(file, vac_doublet_ood, format="extxyz")


    # Neutral
    vacuum_full_neutral_path = os.path.join(vacuum_minimal_path, "neutral")
    if not os.path.exists(vacuum_full_neutral_path):
        os.mkdir(vacuum_full_neutral_path)

    vac_neutral_train = []
    vac_neutral_val = []
    vac_neutral_test = []
    vac_neutral_ood = []

    for item in tqdm(vac_build_full['train']):
        if item.info['charge'] == 0:
            vac_neutral_train.append(item)

    for item in tqdm(vac_build_full['val']):
        if item.info['charge'] == 0:
            vac_neutral_val.append(item)

    for item in tqdm(vac_build_full['test']):
        if item.info['charge'] == 0:
            vac_neutral_test.append(item)

    for item in tqdm(vac_ood_full):
        if item.info['charge'] == 0:
            vac_neutral_ood.append(item)

    file = os.path.join(vacuum_full_neutral_path,'radqm9_65_10_25_sp_vacuum_full_data_20240807_neutral_train.xyz')
    ase.io.write(file, vac_neutral_train, format="extxyz")
    
    file = os.path.join(vacuum_full_neutral_path,'radqm9_65_10_25_sp_vacuum_full_data_20240807_neutral_val.xyz')
    ase.io.write(file, vac_neutral_val, format="extxyz")
    
    file = os.path.join(vacuum_full_neutral_path,'radqm9_65_10_25_sp_vacuum_full_data_20240807_neutral_test.xyz')
    ase.io.write(file, vac_neutral_test, format="extxyz")

    file = os.path.join(vacuum_full_neutral_path,'radqm9_65_10_25_sp_vacuum_full_data_20240807_neutral_ood.xyz')
    ase.io.write(file, vac_neutral_ood, format="extxyz")

    fractions = [.01, .05, .1, .25, .5, .75]

    wtd_full = weight_to_data_ase(vac_build_full["train"])
    cd_full = chunk_data(wtd_full, fractions)
    
    wtd_cs = dict()
    cd_cs = dict()
    for key in vac_train_cs_dict:
        wtd_cs[key] = weight_to_data_ase(vac_train_cs_dict[key])
        cd_cs[key] = chunk_data(wtd_cs[key], fractions)

    wtd_singlet = weight_to_data_ase(vac_singlet_train)
    cd_singlet = chunk_data(wtd_singlet, fractions)

    wtd_doublet = weight_to_data_ase(vac_doublet_train)
    cd_doublet = chunk_data(wtd_doublet, fractions)

    wtd_neutral = weight_to_data_ase(vac_neutral_train)
    cd_neutral = chunk_data(wtd_neutral, fractions)

    for ii, frac in enumerate(fractions):
        chunk_file = os.path.join(vacuum_full_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_full[ii], format="extxyz")
        
        for key in cd_cs:
            chunk_file = os.path.join(vacuum_full_chargespin_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_train_subset_' + key + f'_{frac}.xyz')
            ase.io.write(chunk_file, cd_cs[key][ii], format="extxyz")

        chunk_file = os.path.join(vacuum_full_singlet_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_singlet_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_singlet[ii],format="extxyz")

        chunk_file = os.path.join(vacuum_full_doublet_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_doublet_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_doublet[ii],format="extxyz")
        
        chunk_file = os.path.join(vacuum_full_neutral_path, 'radqm9_65_10_25_sp_vacuum_full_data_20240807_neutral_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_neutral[ii],format="extxyz")

    # Cleanup for memory
    del vac_build_full
    del vac_ood_full
    del vac_train_cs_dict
    del vac_val_cs_dict
    del vac_test_cs_dict
    del vac_ood_cs_dict
    del vac_singlet_train
    del vac_singlet_val
    del vac_singlet_test
    del vac_singlet_ood
    del vac_doublet_train
    del vac_doublet_val
    del vac_doublet_test
    del vac_doublet_ood
    del vac_neutral_train
    del vac_neutral_val
    del vac_neutral_test
    del vac_neutral_ood
    del wtd_full
    del wtd_cs
    del wtd_doublet
    del wtd_neutral

        # SMD data

    wtd = weight_to_data(smd_data)
    sld = length_dict(wtd)

    smd_train_mass = ['152.037'] # EVAN WILL NEED TO ADJUST THE MASSES OF INITIAL POINTS FOR NEW DATA
    smd_test_mass = ['144.092']
    smd_val_mass = ['143.108']

    smd_train = sld['152.037'] # trackers for dataset sizes
    smd_test = sld['144.092']
    smd_val = sld['143.108']

    sld.pop('152.037')
    sld.pop('144.092')
    sld.pop('143.108')

    # Sort the data 
    # data is a dict: mass-># of trajs
    for mass in sld:
        temp_total = smd_train + smd_val + smd_test
        train_ratio = .65 - (smd_train / temp_total)
        test_ratio = .25 - (smd_test / temp_total)
        val_ratio = .1 - (smd_val / temp_total)
        
        if train_ratio > val_ratio and train_ratio > test_ratio:
            smd_train_mass.append(mass)
            smd_train += sld[mass]
        elif val_ratio > train_ratio and val_ratio > test_ratio:
            smd_val_mass.append(mass)
            smd_val += sld[mass]
        else:
            smd_test_mass.append(mass)
            smd_test += sld[mass]

    sld = length_dict(wtd) # you need to call this again yes

    smd_switch = [
        "117.039",
        "116.204",
        "116.204",
        "116.160",
        "113.072",
        "112.216",
        "110.116",
        "109.132",
        "109.092",
        "102.092",
        "102.089",
        "101.065",
        "101.061",
        "100.205",
        "97.117",
        "95.105",
        "95.061",
        "93.089",
        "92.141",
        "88.150",
        "86.050",
        "86.046",
        "85.110",
    ]

    for mass in smd_switch:
        smd_val_mass.append(mass)
        smd_val += sorted_length_dict[mass]
        
        smd_test_mass.remove(mass)
        smd_test -= sorted_length_dict[mass]

    smd_train_data = [wtd[x] for x in smd_train_mass]
    smd_train_data = list(chain.from_iterable(smd_train_data))

    smd_val_data = [wtd[x] for x in smd_val_mass]
    smd_val_data = list(chain.from_iterable(smd_val_data))

    smd_test_data = [wtd[x] for x in smd_test_mass]
    smd_test_data = list(chain.from_iterable(smd_test_data))

    smd_data = {
        'train':smd_train_data,
        'val': smd_val_data,
        'test': smd_test_data
    }

    # Minimal build
    smd_build_minimal = dict()
    for split in data:
        smd_build_minimal[split] = build_minimal_atoms_iterator(smd_data[split], forces="precise_gradient")
        
    create_dataset(smd_build_minimal, 'radqm9_65_10_25_sp_smd_minimal_data_20240807', smd_minimal_path)

    smd_ood_minimal = build_minimal_atoms_iterator(smd_ood, forces="precise_gradient")
    file = os.path.join(smd_minimal_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_ood.xyz')
    ase.io.write(file, smd_ood_minimal, format="extxyz")

    # Charge/spin subsets
    smd_train_cs_dict = {}
    for item in tqdm(smd_build_minimal['train']):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            smd_train_cs_dict[key].append(item)
        except KeyError:
            smd_train_cs_dict[key] = [item]

    smd_val_cs_dict = {}
    for item in tqdm(smd_build_minimal['val']):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            smd_val_cs_dict[key].append(item)
        except KeyError:
            smd_val_cs_dict[key] = [item]

    smd_test_cs_dict = {}
    for item in tqdm(smd_build_minimal['test']):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            smd_test_cs_dict[key].append(item)
        except KeyError:
            smd_test_cs_dict[key] = [item]

    smd_ood_cs_dict = {}
    for item in tqdm(smd_ood_minimal):
        key = str(item.info['charge']) + "_" + str(item.info['spin'])
        try:
            smd_ood_cs_dict[key].append(item)
        except KeyError:
            smd_ood_cs_dict[key] = [item]

    # Split by charge/spin pair
    # Use this for relative energies
    smd_minimal_chargespin_path = os.path.join(smd_minimal_path, "by_charge_spin")
    if not os.path.exists(smd_minimal_chargespin_path):
        os.mkdir(smd_minimal_chargespin_path)

    for key in smd_test_cs_dict:
        file = os.path.join(smd_minimal_chargespin_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_train_'+key+'.xyz')
        ase.io.write(file, smd_train_cs_dict[key], format="extxyz")
        
        file = os.path.join(smd_minimal_chargespin_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_val_'+key+'.xyz')
        ase.io.write(file, smd_val_cs_dict[key], format="extxyz")
        
        file = os.path.join(smd_minimal_chargespin_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_test_'+key+'.xyz')
        ase.io.write(file, smd_test_cs_dict[key], format="extxyz")

        if key in smd_ood_cs_dict:
            file = os.path.join(smd_minimal_chargespin_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_ood_'+key+'.xyz')
            ase.io.write(file, smd_ood_cs_dict[key], format="extxyz")

    # Singlet
    smd_minimal_singlet_path = os.path.join(smd_minimal_path, "singlet")
    if not os.path.exists(smd_minimal_singlet_path):
        os.mkdir(smd_minimal_singlet_path)

    smd_singlet_train = []
    smd_singlet_val = []
    smd_singlet_test = []
    smd_singlet_ood = []

    for item in tqdm(smd_build_minimal['train']):
        if item.info['spin'] == 1:
            smd_singlet_train.append(item)

    for item in tqdm(smd_build_minimal['val']):
        if item.info['spin'] == 1:
            smd_singlet_val.append(item)

    for item in tqdm(smd_build_minimal['test']):
        if item.info['spin'] == 1:
            smd_singlet_test.append(item)

    for item in tqdm(smd_ood_minimal):
        if item.info['spin'] == 1:
            smd_singlet_ood.append(item)

    file = os.path.join(smd_minimal_singlet_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_singlet_train.xyz')
    ase.io.write(file, smd_singlet_train, format="extxyz")
    
    file = os.path.join(smd_minimal_singlet_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_singlet_val.xyz')
    ase.io.write(file, smd_singlet_val, format="extxyz")
    
    file = os.path.join(smd_minimal_singlet_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_singlet_test.xyz')
    ase.io.write(file, smd_singlet_test, format="extxyz")

    file = os.path.join(smd_minimal_singlet_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_singlet_ood.xyz')
    ase.io.write(file, smd_singlet_ood, format="extxyz")

    # Doublet
    smd_minimal_doublet_path = os.path.join(smd_minimal_path, "doublet")
    if not os.path.exists(smd_minimal_doublet_path):
        os.mkdir(smd_minimal_doublet_path)

    smd_doublet_train = []
    smd_doublet_val = []
    smd_doublet_test = []
    smd_doublet_ood = []

    for item in tqdm(smd_build_minimal['train']):
        if item.info['spin'] == 2:
            smd_doublet_train.append(item)

    for item in tqdm(smd_build_minimal['val']):
        if item.info['spin'] == 2:
            smd_doublet_val.append(item)

    for item in tqdm(smd_build_minimal['test']):
        if item.info['spin'] == 2:
            smd_doublet_test.append(item)

    for item in tqdm(smd_ood_minimal):
        if item.info['spin'] == 2:
            smd_doublet_ood.append(item)

    file = os.path.join(smd_minimal_doublet_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_doublet_train.xyz')
    ase.io.write(file, smd_doublet_train, format="extxyz")
    
    file = os.path.join(smd_minimal_doublet_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_doublet_val.xyz')
    ase.io.write(file, smd_doublet_val, format="extxyz")
    
    file = os.path.join(smd_minimal_doublet_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_doublet_test.xyz')
    ase.io.write(file, smd_doublet_test, format="extxyz")

    file = os.path.join(smd_minimal_doublet_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_doublet_ood.xyz')
    ase.io.write(file, smd_doublet_ood, format="extxyz")


    # Neutral
    smd_minimal_neutral_path = os.path.join(smd_minimal_path, "neutral")
    if not os.path.exists(smd_minimal_neutral_path):
        os.mkdir(smd_minimal_neutral_path)

    smd_neutral_train = []
    smd_neutral_val = []
    smd_neutral_test = []
    smd_neutral_ood = []

    for item in tqdm(smd_build_minimal['train']):
        if item.info['charge'] == 0:
            smd_neutral_train.append(item)

    for item in tqdm(smd_build_minimal['val']):
        if item.info['charge'] == 0:
            smd_neutral_val.append(item)

    for item in tqdm(smd_build_minimal['test']):
        if item.info['charge'] == 0:
            smd_neutral_test.append(item)

    for item in tqdm(smd_ood_minimal):
        if item.info['charge'] == 0:
            smd_neutral_ood.append(item)

    file = os.path.join(smd_minimal_neutral_path,'radqm9_65_10_25_sp_smd_minimal_data_20240807_neutral_train.xyz')
    ase.io.write(file, smd_neutral_train, format="extxyz")
    
    file = os.path.join(smd_minimal_neutral_path,'radqm9_65_10_25_sp_smd_minimal_data_20240807_neutral_val.xyz')
    ase.io.write(file, smd_neutral_val, format="extxyz")
    
    file = os.path.join(smd_minimal_neutral_path,'radqm9_65_10_25_sp_smd_minimal_data_20240807_neutral_test.xyz')
    ase.io.write(file, smd_neutral_test, format="extxyz")

    file = os.path.join(smd_minimal_neutral_path,'radqm9_65_10_25_sp_smd_minimal_data_20240807_neutral_ood.xyz')
    ase.io.write(file, smd_neutral_ood, format="extxyz")

    fractions = [.01, .05, .1, .25, .5, .75]

    wtd_minimal = weight_to_data_ase(smd_build_minimal["train"])
    cd_minimal = chunk_data(wtd_minimal, fractions)
    
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
        chunk_file = os.path.join(smd_minimal_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_minimal[ii], format="extxyz")
        
        for key in cd_cs:
            chunk_file = os.path.join(smd_minimal_chargespin_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_train_subset_' + key + f'_{frac}.xyz')
            ase.io.write(chunk_file, cd_cs[key][ii], format="extxyz")

        chunk_file = os.path.join(smd_minimal_singlet_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_singlet_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_singlet[ii],format="extxyz")

        chunk_file = os.path.join(smd_minimal_doublet_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_doublet_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_doublet[ii],format="extxyz")
        
        chunk_file = os.path.join(smd_minimal_neutral_path, 'radqm9_65_10_25_sp_smd_minimal_data_20240807_neutral_train_subset_' + f'{frac}.xyz')
        ase.io.write(chunk_file, cd_neutral[ii],format="extxyz")

    # Cleanup for memory
    del smd_build_minimal
    del smd_ood_minimal
    del smd_train_cs_dict
    del smd_val_cs_dict
    del smd_test_cs_dict
    del smd_ood_cs_dict
    del smd_singlet_train
    del smd_singlet_val
    del smd_singlet_test
    del smd_singlet_ood
    del smd_doublet_train
    del smd_doublet_val
    del smd_doublet_test
    del smd_doublet_ood
    del smd_neutral_train
    del smd_neutral_val
    del smd_neutral_test
    del smd_neutral_ood
    del wtd_minimal
    del wtd_cs
    del wtd_doublet
    del wtd_neutral

    # Full build
    smd_build_full = dict()
    for split in data:
        smd_build_full[split] = build_atoms_iterator(smd_data[split], forces="precise_gradient")
        
    create_dataset(build_full, 'radqm9_65_10_25_sp_smd_full_data_20240807', smd_full_path)

    smd_ood_full = build_atoms_iterator(smd_ood, forces="precise_gradient")
    file = os.path.join(smd_full_path, 'radqm9_65_10_25_sp_smd_full_data_20240807_ood.xyz')
    ase.io.write(file, smd_ood_full, format="extxyz")

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
    smd_full_neutral_path = os.path.join(smd_minimal_path, "neutral")
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
    
    # TODO: 
    # - Calculate relative energies & regenerate