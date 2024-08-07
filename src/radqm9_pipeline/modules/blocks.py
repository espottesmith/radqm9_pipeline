from monty.serialization import loadfn
from tqdm import tqdm
import collections
from glob import glob
import numpy as np
import ase
import ase.io

from radqm9_pipeline.elements import read_elements

import os
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

elements_dict = read_elements(os.path.join(CUR_DIR, 'elements.pkl'))

def merge_data(folder: str):
    """
    Load and merge the data into a single list from a folder of json files.
    """
    files = glob(folder+'/*')
    merged_data = []
    
    for file in tqdm(files):
        data = loadfn(file)
        merged_data+=data
    
    return merged_data

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
        formatted_data['mulliken_partial_charges'] = item['mulliken_partial_charges']
        formatted_data['mulliken_partial_spins'] = item['mulliken_partial_spins']
        formatted_data['resp_partial_charges'] = item['resp_partial_charges']
        dataset.append(formatted_data)
    return dataset


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
               
                    try:
                        item['mulliken_partial_spins'] = list(itertools.chain.from_iterable(item['mulliken_partial_spins']))
                    except TypeError:
                        # We will resolve weird spin data later in the pipeline
                        pass
                    resolved_data.append(item)   
                else:
                    bad_data.append(item)
        except TypeError:
            non_spin_broken.append(item)
        
    return resolved_data, unequal_data, bad_data, non_spin_broken, spin_broken

def add_unique_id(data: list):
    for item in tqdm(data):
        item['charge_spin'] = str(item['charge']) + str(item['spin'])
        item['mol_cs'] = str(item['mol_id']) + str(item['charge_spin'])

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
        
def flatten_filter(data: dict):
    """
    Flatten bucket (dict) to list
    """
    data_to_be_parsed = []
    for mol_id in tqdm(data):
        for pair in data[mol_id]:
            data_to_be_parsed.append(pair)
    return data_to_be_parsed    

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
            

            
def resolve_mulliken_partial_spins(data: list):
    for item in tqdm(data):
        if item['charge_spin']=='01':
            if item['mulliken_partial_spins'] is None or None in item['mulliken_partial_spins']:
                item['mulliken_partial_spins']=[0]

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
                dipole_moments_resp = [pair['calc_resp_dipole_moments'][0], pair['calc_resp_dipole_moments'][-1]]

                pair['geometries'] = geometries
                pair['energies'] = energies
                pair['gradients'] = grads
                pair['mulliken_partial_charges'] = mulliken_partial_charges
                pair['mulliken_partial_spins'] = mulliken_partial_spins
                pair['resp_partial_charges'] = resp_partial_charges
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
                pair['calc_resp_dipole_moments'] = dipole_moments_resp
        except ValueError:
            bad.append(pair)
    return bad
    
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
            if next_item:
                break
        if not next_item:
            good.append(item)
                            
    return good, bad, weird

def filter_charges(data: list, charge: list):
    clean = []
    bad = []
    for item in data:
        if item['charge'] not in charge:
            clean.append(item)
        else:
            bad.append(item)
    return clean, bad

def convert_energy(data: list):
    for item in tqdm(data):
        energy = item['energy']
        item['energy'] = [x*27.2114 for x in energy]

def convert_forces(data: list):
    for item in tqdm(data):
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
        for traj_point in item['geometries'][1:]: # Ignore the first is in the filter_broken_graphs
            graph = build_graph(item['species'], traj_point)
            connected = nx.is_connected(graph.graph.to_undirected())
            if not connected:
                broken.append(item)
                break
        if connected:
            good.append(item)

    return good, broken



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

def get_molecule_weight(data: list):
    """
    The method takes in a list of data (either trajectories or single points) and sorts into a distribution
    dictionary. The keys are the species/formula and the value of each key is the weight. appearing number of times the species
    appears in the dataset.
    """
    dict_dist = {}
    for item in tqdm(data):
        species_num = []
        species=''.join((sorted(item['species'])))
        
        for element in item['species']:
            species_num.append(elements_dict[element])

        species_sum = sum(species_num)
        try:
            dict_dist[species].append(species_sum)
            dict_dist[species] = [dict_dist[species][0]]*len(dict_dist[species])
        except KeyError:
            dict_dist[species] = [species_sum]
        
    return dict_dist

def molecule_weight(data: list, weight_dict):
    """
    This method takes in data and assigns the mass.
    Python does a weird thing floats e.g., {126.15499999999993, 126.15499999999994}, having this and
    get_molecule_weight gurantees that species that are the same are not being assigned different weights.
    """
    for item in tqdm(data):
        weight = weight_dict[''.join((sorted(item['species'])))][0]
        item['molecule_mass'] = weight
        
def weight_to_data(data: list):
    """
    This method buckets the data by the mass such that the dict key is the mass and the values are the data
    points.
    """
    dict_data = {}
    for item in tqdm(data):
        try:
            dict_data[item['molecule_mass']].append(item)
        except KeyError:
            dict_data[item['molecule_mass']] = [item]
    return dict_data

def length_dict(data: dict):
    """
    This method takes in the output of weight_to_data and returns a dictionary that is sorted from largest
    to smallest mass. The keys are the mass and the values are the number of appearances.
    """
    length_dict = {key: len(value) for key, value in data.items()}
    sorted_length_dict = {k: length_dict[k] for k in sorted(length_dict, reverse=True)}
    
    return sorted_length_dict

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
    for i in range(len(data['geometries'])):
        atoms = ase.atoms.Atoms(
            symbols=data['species'],
            positions=data['geometries'][i]
        )
        atoms.arrays['mulliken_partial_charges']=np.array(data['mulliken_partial_charges'][i])
        atoms.arrays['mulliken_partial_spins']=np.array(data['mulliken_partial_spins'][i])
        atoms.arrays['resp_partial_charges']=np.array(data['resp_partial_charges'][i])
        atoms.info['calc_resp_dipole_moments']=np.array(data['calc_resp_dipole_moments'][i])
        
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
            if data['charge_spin'] == '0,1':
                atoms.info['position_type'] = 'end'
            else:
                atoms.info['position_type'] = 'middle'
        if i == 2:
            atoms.info['position_type'] = 'end'
        atom_list.append(atoms)
    return atom_list



def build_minimal_atoms_iterator(data: list,
                         train=False):
    """
    This method assumes the data has been validated. This will create ASE atoms to be written.
    
    The input needs to be a list of lists that contain the event dictionaries. Each inner list needs to represent all the events for a single
    mol_id.
    """
    data_set=[]
    for point in tqdm(data):
        atoms=build_minimal_atoms(point, energy='energies', forces='gradients', charge='charge', spin='spin', train=train)
        data_set+=atoms
    return data_set


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
    
    
def get_molecule_weight_ase(data: list):
    dict_dist = {}
    data_dict = {}
    for item in tqdm(data):
        species_num = []
        species=''.join((sorted(item.get_chemical_symbols())))
        
        for element in item.get_chemical_symbols():
            species_num.append(elements_dict[element])

        species_sum = sum(species_num)
        try:
            dict_dist[species].append(species_sum)
            # python does a weird thing floats e.g., {126.15499999999993, 126.15499999999994}
            dict_dist[species] = [dict_dist[species][0]]*len(dict_dist[species])
        except KeyError:
            dict_dist[species] = [species_sum]
        
    return dict_dist

def get_molecule_weight_ase(data: list):
    dict_dist = {}
    data_dict = {}
    for item in tqdm(data):
        species_num = []
        species=''.join((sorted(item.get_chemical_symbols())))
        
        for element in item.get_chemical_symbols():
            species_num.append(elements_dict[element])

        species_sum = sum(species_num)
        try:
            dict_dist[species].append(species_sum)
            # python does a weird thing floats e.g., {126.15499999999993, 126.15499999999994}
            dict_dist[species] = [dict_dist[species][0]]*len(dict_dist[species])
        except KeyError:
            dict_dist[species] = [species_sum]
        
    return dict_dist

def weight_to_data_ase(data: list):
    dict_data = {}
    for item in tqdm(data):
        try:
            dict_data[item.info['weight']].append(item)
        except KeyError:
            dict_data[item.info['weight']] = [item]
    return dict_data

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
                            # print(sample_size)
                            # print(len(foo_data[key]))
                            add_on = foo_data[key][:sample_size]
                            chunk_data += add_on
                            # print(len(foo_data[key][:sample_size])/len(foo_data[key]))
                            foo_data[key] = foo_data[key][sample_size:]
                            counter += len(add_on)
                            if counter >= sizes[i]-sizes[i-1]:
                                break
                else:
                    # print(counter)
                    # print(sizes[i])
                    # print(sizes[i]/total)
                    # print(len(chunk_data)/total)
                    break
                # if counter > total:
                #     print('bad')
                #     break

            return_data[i] = chunk_data + return_data[i-1]
    return return_data    

def relative_energies(data: list, stats: dict):
    for item in tqdm(data):
        key = str(item.info['charge'])+str(item.info['spin'])
        lookup_sum = 0
        for num in item.arrays['numbers']:
            lookup_sum += eval(stats[key]['atomic_energies'])[num]
        
        rel = item.info['energy'] - lookup_sum
        item.info['relative_energy'] = rel
        
def relative_single_json_energies(data: list, stats: dict):
    for item in tqdm(data):
        lookup_sum = 0
        for num in item.arrays['numbers']:
            lookup_sum += eval(stats['atomic_energies'])[num]
        
        rel = item.info['total_energy'] - lookup_sum
        item.info['relative_energy'] = rel
        
def re_build_atoms_rel(data: dict,
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
    for item in tqdm(data):
        atoms = ase.atoms.Atoms(
            numbers=item.arrays['numbers'],
            positions=item.arrays['positions']
        )
        atoms.info['total_energy'] = item.info['total_energy']
        atoms.info['relative_energy'] = item.info['relative_energy']
        atoms.info['mol_id'] = item.info['mol_id']
        atoms.arrays['forces'] = np.array(item.arrays['forces'])
        atoms.info['charge'] =  item.info['charge']
        atoms.info['spin'] =  item.info['spin'] 
        atoms.info['position_type'] = item.info['position_type']
        atoms.arrays['mulliken_partial_charges']=np.array(item.arrays['mulliken_partial_charges'])
        atoms.arrays['mulliken_partial_spins']=np.array(item.arrays['mulliken_partial_spins'])
        atoms.arrays['resp_partial_charges']=np.array(item.arrays['resp_partial_charges'])
        atoms.info['calc_resp_dipole_moments']=np.array(item.info['calc_resp_dipole_moments'])
        
        atom_list.append(atoms)
    return atom_list

