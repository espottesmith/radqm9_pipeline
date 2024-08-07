import ase
import networkx as nx
import os
from monty.serialization import loadfn
from glob import glob
import time
from tqdm import tqdm
import collections
import numpy as np
import matplotlib.pyplot as plt
import ast
import h5py


from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, CovalentBondNN
from pymatgen.util.graph_hashing import weisfeiler_lehman_graph_hash

from radqm9_pipeline.elements import read_elements
from radqm9_pipeline.modules import merge_data

elements_dict = read_elements('/pscratch/sd/m/mavaylon/sam_ldrd/radqm9_pipeline/src/radqm9_pipeline/modules/elements.pkl')

import sys
from itertools import chain
from pathlib import Path

from maggma.stores.mongolike import MongoStore


# Single-point/force information
force_store = MongoStore(database="thermo_chem_storage",
                           collection_name="radqm9_trajectories",
                           username="thermo_chem_storage_ro",
                           password="2322jj2dsd",
                           host="mongodb07.nersc.gov",
                           port=27017,
                           key="molecule_id")
force_store.connect()

data = []
for item in tqdm(force_store.query({})):
    data.append(item)

    
def filter_features(data: list):
    dataset = []
    for item in tqdm(data):
        formatted_data={}
        formatted_data['mol_id'] = item['molecule_id']
        formatted_data['species'] = item['species']
        formatted_data['charge'] = item['charge'] 
        formatted_data['spin'] = item['spin_multiplicity']
        formatted_data['geometries'] = item['geometries']
        formatted_data['energy'] = item['energies']
        formatted_data['gradients'] = item['forces']
        # formatted_data['dipole_moments'] = item['dipole_moments']
        # formatted_data['resp_dipole_moments'] = item['resp_dipole_moments']
        formatted_data['mulliken_partial_charges'] = item['mulliken_partial_charges']
        formatted_data['mulliken_partial_spins'] = item['mulliken_partial_spins']
        formatted_data['resp_partial_charges'] = item['resp_partial_charges']
        dataset.append(formatted_data)
    return dataset

dataset = filter_features(data)

import itertools

import itertools

def resolve_trajectories(data: list):
    resolved_data = []
    unequal_data = []
    bad_data = []
    non_spin_broken = []
    spin_broken = []
    for item in tqdm(data):
        try:
            feat = [len(item['geometries']), len(item['gradients']), len(item['energy'])]

            feat_set = set(feat)
            if len(feat_set) !=1 :
                unequal_data.append([item, feat])
            else:
                len_geo = len(item['geometries'])
                if len_geo==1:
                    resolved_data.append(item)
                elif len_geo > 1:
                    item['geometries'] = list(itertools.chain.from_iterable(item['geometries']))
                    item['gradients'] = list(itertools.chain.from_iterable(item['gradients']))
                    item['energy'] = list(itertools.chain.from_iterable(item['energy']))
                    item['mulliken_partial_charges'] = list(itertools.chain.from_iterable(item['mulliken_partial_charges']))
                    item['resp_partial_charges'] = list(itertools.chain.from_iterable(item['resp_partial_charges']))
                    # item['resp_dipole_moments'] = list(itertools.chain.from_iterable(item['resp_dipole_moments']))
                    # item['dipole_moments'] = list(itertools.chain.from_iterable(item['dipole_moments']))
                    
                    try:
                        item['mulliken_partial_spins'] = list(itertools.chain.from_iterable(item['mulliken_partial_spins']))
                    except TypeError:
                        # We wil resolve weird spin data later in the pipeline
                        pass
                    resolved_data.append(item)   
                else:
                    bad_data.append(item)
        except TypeError:
            non_spin_broken.append(item)
        
    return resolved_data, unequal_data, bad_data, non_spin_broken, spin_broken


r_data, u_data, b_data, non_spin_data, spin_data = resolve_trajectories(dataset)

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
        # if len(item['resp_dipole_moments']) == 1:
        #     item['resp_dipole_moments'] = item['resp_dipole_moments'][0]
        # if len(item['dipole_moments']) == 1:
        #     item['dipole_moments'] = item['dipole_moments'][0]

dimension(r_data)



def add_unique_id(data: list):
    for item in tqdm(data):
        item['charge_spin'] = str(item['charge']) + str(item['spin'])
        item['mol_cs'] = str(item['mol_id']) + str(item['charge_spin'])
        
add_unique_id(r_data)

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

        
generate_resp_dipole(r_data)

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

def resolve_mulliken_partial_spins(data: list):
    for item in tqdm(data):
        if item['charge_spin']=='01':
            if item['mulliken_partial_spins'] is None or None in item['mulliken_partial_spins']:
                item['mulliken_partial_spins']=[0]

resolve_mulliken_partial_spins(r_data)                
                
def filter_data(data: list):
    good = []
    filtered = []
    for item in data:
        if item['charge_spin'] != '01':
            if len(item['gradients']) < 2:
                filtered.append(item)
            else:
                good.append(item)
        else:
            good.append(item)
    
    return good, filtered
            
g_data, f_data = filter_data(r_data)


def force_magnitude_filter(cutoff: float,
                           data: list):
    """
    This method returns both data that meets the cuttoff value and data that is equal to or above the cuttoff value.
    If this is run before downsampling, it removes the entire data point trajectory.
    
    Returns: lists
    """
    good = []
    bad = []
    weird = []
    for item in tqdm(data):
        forces = item['gradients']
        for path_point in forces:
            next_item = False
            for atom in path_point:
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
                    # weird.append(item)
                    # next_item = True
            if next_item:
                break
        if not next_item:
            good.append(item)
                            
    return good, bad, weird


g_data, b_data, w_data = force_magnitude_filter(cutoff=10.0, data=g_data)



def sparse_trajectory(data: list):
    """
    This takes the cleaned data and will sparsifiy the optimization trajectories. How this is done will depend on the
    charge_spin pair:
    - Neutral Singlet (0,1): First and Last
    - Other: First, Last, and structure with the highest molecular force other than the First.
    
    Note: Molecular Force is just the average of the force magnitudes of each atom in the molecule:
    """
    bad=[]

    for pair in tqdm(data):
        try:
            if pair['charge_spin'] == '01':
                geometries = [pair['geometries'][0], pair['geometries'][-1]]
                energies = [pair['energy'][0], pair['energy'][-1]]
                grads = [pair['gradients'][0], pair['gradients'][-1]]
                mulliken_partial_charges = [pair['mulliken_partial_charges'][0], pair['mulliken_partial_charges'][-1]]
                mulliken_partial_spins = [pair['mulliken_partial_spins'][0], pair['mulliken_partial_spins'][-1]]
                resp_partial_charges = [pair['resp_partial_charges'][0], pair['resp_partial_charges'][-1]]
                # dipole_moments = [pair['dipole_moments'][0][0], pair['dipole_moments'][0][-1]]
                dipole_moments_resp = [pair['calc_resp_dipole_moments'][0], pair['calc_resp_dipole_moments'][-1]]

                pair['geometries'] = geometries
                pair['energies'] = energies
                pair['gradients'] = grads
                pair['mulliken_partial_charges'] = mulliken_partial_charges
                pair['mulliken_partial_spins'] = mulliken_partial_spins
                pair['resp_partial_charges'] = resp_partial_charges
                # pair['dipole_moments'] = dipole_moments
                pair['calc_resp_dipole_moments'] = dipole_moments_resp
            else:
                force_dict = average_force_trajectory(pair)
                max_index = max(force_dict, key=force_dict.get)

                geometries = [pair['geometries'][0], pair['geometries'][max_index], pair['geometries'][-1]]
                energies = [pair['energy'][0], pair['energy'][max_index], pair['energy'][-1]]
                grads = [pair['gradients'][0], pair['gradients'][max_index], pair['gradients'][-1]]
                mulliken_partial_charges = [pair['mulliken_partial_charges'][0], pair['mulliken_partial_charges'][max_index], pair['mulliken_partial_charges'][-1]]
                mulliken_partial_spins = [pair['mulliken_partial_spins'][0], pair['mulliken_partial_spins'][max_index], pair['mulliken_partial_spins'][-1]]
                resp_partial_charges = [pair['resp_partial_charges'][0], pair['resp_partial_charges'][max_index], pair['resp_partial_charges'][-1]]
                dipole_moments_resp = [pair['calc_resp_dipole_moments'][0], pair['calc_resp_dipole_moments'][max_index], pair['calc_resp_dipole_moments'][-1]]


                pair['geometries'] = geometries
                pair['energies'] = energies
                pair['gradients'] = grads
                pair['mulliken_partial_charges'] = mulliken_partial_charges
                pair['mulliken_partial_spins'] = mulliken_partial_spins
                pair['resp_partial_charges'] = resp_partial_charges
                # pair['dipole_moments'] = dipole_moments
                pair['calc_resp_dipole_moments'] = dipole_moments_resp
        except ValueError:
            bad.append(pair)
    return bad
sparse_trajectory(g_data)

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
        if item['charge_spin'] == '01':
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

filtered_good, broken =filter_broken_graphs(g_data)

import h5py
with h5py.File('...', 'w') as file:
    g1 =file.create_group('clean_data')
    raw=[]
    for item in tqdm(filtered_good):
        raw.append(str(item))
    g1.create_dataset('data',data=raw)
    
with h5py.File('...', 'w') as file:
    g1 =file.create_group('broken_data')
    raw=[]
    for item in tqdm(broken):
        raw.append(str(item))
    g1.create_dataset('data',data=raw)