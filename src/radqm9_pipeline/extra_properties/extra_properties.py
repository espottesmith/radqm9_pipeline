import os
import collections
from glob import glob
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

import ase
import ase.io

from maggma.core import Store
from monty.serialization import loadfn

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph

from radqm9_pipeline.elements import read_elements


CUR_DIR = os.path.dirname(os.path.realpath(__file__))

elements_dict = read_elements(os.path.join(CUR_DIR, 'elements.pkl'))

# "total_dipole",
# "resp_total_dipole",
# "quadrupole_moment",
# "octopole_moment",
# "hexadecapole_moment",

def create_properties_dataset(
    store: Store,
    query: Optional[Dict] = None,
    filter_broken_graphs: bool = True,
    solvents: List[str] = [
        "NONE",
        "SOLVENT=WATER"
    ]
    required_opt_fields: List[str] = [
        "zero_point_energy",
        "total_enthalpy",
        "total_entropy",
        "rotational_enthalpy",
        "rotational_entropy",
        "vibrational_enthalpy",
        "vibrational_entropy",
        "frequencies",
        "frequency_modes",
        "ir_intensities",
        "ir_activities",
        "raman_intensities",
        "raman_activities",
    ],
    required_sp_fields: List[str] = [
        "free_energy",
        "electron_affinity",
        "ionization_energy",
    ],
    additional_opt_fields: List[str] = list(),
    additional_sp_fields: List[str] = [
        "reduction_free_energy",
        "oxidation_free_energy"
    ]
):
    """
    Create XYZ-based dataset for additional properties

    Args:
        store (Store): Database to query to obtain summary documents
        query (Dict): Database query. Default is an empty dict
        filter_broken_graphs (bool): Should "molecules" with multiple fragments be included in this dataset?
            Default is True, meaning that molecules with multiple fragments will be filtered out.
        solvents (List[str]): List of solvents to check for. Default is ["NONE", "SOLVENT=WATER"]
        required_opt_fields (List[str]): List of fields derived from optimization/frequency calculations that are
            reqiured for all entries. Defaults include:
                - "zero_point_energy"
                - "total_enthalpy"
                - "total_entropy"
                - "rotational_enthalpy",
                - "rotational_entropy"
                - "vibrational_enthalpy"
                - "vibrational_entropy"
                - "frequencies"
                - "frequency_modes"
                - "ir_intensities"
                - "ir_activities"
                - "raman_intensities"
                - "raman_activities"
        required_sp_fields (List[str]): List of fields derived from single-point energy and force calculations that
            are required for all entries. Defaults include:
                - "free_energy"
                - "electron_affinity"
                - "ionization_energy"
        additional_opt_fields (List[str]): List of fields derived from optimization/frequency calculations that are
            optional. Default is an empty list.
        additional_sp_fields (List[str]): List of fields derived from single-point energy and force calculations that
            are optional. Defaults include:
                - "reduction_free_energy"
                - "oxidation_free_energy"
    """

    total = store.count(criteria=query)

    broken = list()
    good = list()

    weight_dist = dict()
    weight_dict = dict()

    with tqdm(desc="Initial iterator over summary documents", total=total) as pbar:
        for ii, molecule in enumerate(store.query(criteria=query)):
            mol_id = molecule["molecule_id"]

            molecule["charge_spin"] = ",".join([str(molecule["charge"]), str(molecule["spin_multiplicity"])])

            mol = Molecule.from_dict(molecule["molecule"])
            species = [str(e) for e in mol.species]

            species_num = list()
            species_sorted = ''.join(sorted(set(species)))
            for element in species:
                species_num.append(elements_dict[element])

            molecule['weight_tag'] = round(sum(species_num))

            mg = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())

            if not nx.is_connected(mg.graph.to_undirected()):
                broken.append(molecule["molecule_id"])
            else:

                try:
                    weight_dist[round(sum(species_num))]+=1
                except KeyError:
                    weight_dist[round(sum(species_num))]=1
                    
                try:
                    weight_dict[str(sum(species_num))+'_'+species_sorted].append(mol_id)
                except KeyError:
                    weight_dict[str(sum(species_num))+'_'+species_sorted] = [mol_id]

            pbar.update()


def train_val_test_split(bucket: dict,
                         train_size: float,
                         val_size: float):
    """
    This method takes in the output from mol_id_weight_bins.
    This method will sample from each key-value pair from the input dict based on the train_size, val_size.
    The method requires a validation set, but the user can combine it with test if they so choose.
    """
    
    weight_dict, weight_dist, bucket = __mol_id_weight_bins(bucket)
    
    train_marker = train_size
    val_marker = train_size + val_size
    
    split={}

    import random
    random.seed(10)
    for strata in tqdm(weight_dict):
        random.shuffle(weight_dict[strata])
        train_index = round(len(weight_dict[strata])*train_marker)
        val_index = round(len(weight_dict[strata])*val_marker)
        # print(len(weight_dict[strata]))
        # print(train_index)
        # print(val_index)
        # break
              
        
        try:
            train_split = (weight_dict[strata][:train_index])
            val_split = (weight_dict[strata][train_index:val_index+1])
            test_split = (weight_dict[strata][val_index+1:])
            
            if len(test_split)> len(val_split):
                print('bleh')
                return [weight_dict[strata], train_split, val_split, test_split, train_index, val_index+1]
            
            split['train']+=train_split
            split['val']+=val_split
            split['test']+=test_split
            
            
        except KeyError:
            split['train'] = weight_dict[strata][:train_index]
            split['val'] = weight_dict[strata][train_index:val_index+1]
            split['test'] = weight_dict[strata][val_index+1:]
    train_data = [bucket[i] for i in split['train']]
    val_data = [bucket[i] for i in split['val']]
    test_data = [bucket[i] for i in split['test']]
                 
    split['train']=train_data
    split['val']=val_data
    split['test']=test_data
    
    return split, weight_dist

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def flatten(data):
    return [x for y in data for x in y]

def tagger(train_data, dist):
    
    # Flatten train_data into a pool of data points
    flat = [x for y in train_data for x in y] 
    
    # Find all the weights that appear less than 20 occurances
    # cutoff_bin = [x for x in dist if .05*dist[x]<1]
    
    bins = {}
    chunked_training = {}
    
    for point in flat:
        try:
            bins[point['weight_tag']].append(point['mol_id'])
        except KeyError:
            bins[point['weight_tag']] = [point['mol_id']]
        
    
    mol_id_tag = {}
    chunks = []
    for weight_group in tqdm(bins):
        tag_groups = list(split(list(set(bins[weight_group])),20))
        chunk = 0
        for group in tag_groups:
            for mol_id in group:
                chunks.append(chunk)
                mol_id_tag[mol_id] = chunk
            chunk += 5
    print(set(chunks))
    
    for point in tqdm(flat):
        if point['mol_id'] in mol_id_tag:
            point['chunk'] = mol_id_tag[point['mol_id']] 
            try:
                chunked_training[mol_id_tag[point['mol_id']]].append(point)
            except KeyError:
                chunked_training[mol_id_tag[point['mol_id']]] = [point]
    return flat, chunked_training

def charge_filter(charges: list, data):
    """
    Takes both a list of charges to filter by and a dataset that is a list of data points.
    """
    filtered_data = []
    for point in tqdm(data):
        if point['charge'] in charges:
            filtered_data.append(point)
    
    return filtered_data

def chunk_train_multiple(data: list, percentage: list):
    """
    Percentage: list of percentages e.g., [.05, .1, .25, .5, .75]
    """
    mol_id_bucket = {}
    for point in tqdm(data):
        try:
            mol_id_bucket[point['mol_id']].append(point)
        except KeyError:
            mol_id_bucket[point['mol_id']] = [point]
    
    
    elements_dict = read_elements('/pscratch/sd/m/mavaylon/sam_ldrd/radqm9_pipeline/src/radqm9_pipeline/modules/elements.pkl')
    
    ##################
    # Create a weight dictionary such that the keys are the unique weights using atomic mass of the molecule
    # as a float and the values are list of mol_ids that correspond to said weight. 
    ##################
    weight_dict = {}
    for point in tqdm(data):
        species = point['species']
        species_num = 0
        for element in species:
            species_num+=elements_dict[element]
        try:
            weight_dict[str(species_num)].append(point['mol_id'])
            weight_dict[str(species_num)] = list(set(weight_dict[str(species_num)]))
        except KeyError:
            weight_dict[str(species_num)] = [point['mol_id']]
    
    ##################
    # Calculate total data points
    ##################
    total=0
    for pair in tqdm(data):
        total+=len(pair['geometries'])
    print(total)
    
    ##################
    # Calculate the size for the data chunk for each percentage
    ##################
    sizes = []
    for item in percentage:
        temp_size = round(total*item)
        sizes.append(temp_size)
    
    ##################
    # Get the chunked mol_ids for each size
    ##################
    chunks = []
    count = 0
    
    for size in tqdm(sizes):
        print("size:", size)
        chunked_mol_id_data = []
        # while count<size:
        for i in tqdm(range(total)):
            if count<size:
                for key in weight_dict:
                    if len(weight_dict[key])!=0:
                        _id = weight_dict[key][0]
                        for point in mol_id_bucket[_id]:
                            count += len(point['geometries'])
                        # print(count)

                        chunked_mol_id_data.append(_id)
                        weight_dict[key] = weight_dict[key][1:]

                    else:
                        pass
                weight_dict = {k: v for (k,v) in weight_dict.items() if len(v)!=0}

            else:
                break
        print('count:', count)
        # print(len(chunked_mol_id_data))
        chunks.append(chunked_mol_id_data)
        
        for item in chunks:
            print(len(item))
        
    chunked_data = []
    for chunk_set in tqdm(chunks):
        chunk = []
        for item in tqdm(chunk_set):
            chunk+=mol_id_bucket[item]
        chunked_data.append(chunk)
    
    return chunked_data, weight_dict

def build_atoms(data: dict,
                energy: str = None,
                forces: str = None,
                charge:str = None,
                spin:str = None,
                train = False) -> ase.Atoms:
    """ 
    Populate Atoms class with atoms in molecule.
        atoms.info : global variables
        atoms.array : variables for individual atoms
        
    Both "energy" and "forces" are the dict strings in data.
    """
    atom_list = []
    try:
        for i in range(len(data['geometries'])):
            atoms = ase.atoms.Atoms(
                symbols=data['species'],
                positions=data['geometries'][i]
            )
            if energy is not None:
                atoms.info['energy'] = data[energy][i]
            if forces is not None:
                atoms.arrays['forces'] = np.array(data[forces][i])
            if charge is not None:
                 atoms.info['charge'] = data[charge]
            if spin is not None:
                atoms.info['spin'] = data[spin]
            if train:
                atoms.info['chunk'] = data['chunk']
            if i == 0:
                atoms.info['position_type'] = 'start'
            atoms.info['mol_id'] = data['mol_id']
            if i == 1:
                if data['charge_spin'] == '0,1':
                    atoms.info['position_type'] = 'end'
                else:
                    atoms.info['position_type'] = 'middle'
            if i == 2:
                atoms.info['position_type'] = 'end'
            atom_list.append(atoms)
    except IndexError:
        print(i)
        print(data['mol_id'])

def build_atoms_iterator(data: list,
                         train=False):
    """
    This method assumes the data has been validated. This will create ASE atoms to be written.
    
    The input needs to be a list of lists that contain the event dictionaries. Each inner list needs to represent all the events for a single
    mol_id.
    """
    data_set=[]
    for point in tqdm(data):
        atoms=build_atoms(point, energy='energies', forces='gradients', charge='charge', spin='spin', train=train)
        data_set+=atoms
    return data_set

def build_manager(data: dict, weight_dist, train):
    """
    Manage building atoms for train/val/test splits
    """
    
    data['train'] = tagger(data['train'], weight_dist)
    data['val'] = flatten(data['val'])
    data['test'] = flatten(data['test'])
    
    build = {}
    for split in data:
        if split == 'train':
            build[split] = build_atoms_iterator(data[split], train=train)
        else:
            build[split] = build_atoms_iterator(data[split])
    return build

def create_dataset(data: dict,
                   file_name:str,
                   path:str):
    """
    This method will handle the I/O for writing the data to xyz files to the path provided.
    """
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']
    
    train_file = os.path.join(path,file_name+'_train.xyz')
    ase.io.write(train_file, train_data,format="extxyz")
     
    val_file = os.path.join(path,file_name+'_val.xyz')
    ase.io.write(val_file, val_data,format="extxyz")
    
    test_file = os.path.join(path,file_name+'_test.xyz')
    ase.io.write(test_file, test_data,format="extxyz")