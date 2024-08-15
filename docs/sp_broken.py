import ase

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

import networkx as nx
from tqdm import tqdm
import ast

from radqm9_pipeline.modules import read_elements

elements_dict = read_elements('/pscratch/sd/m/mavaylon/sam_ldrd/radqm9_pipeline/src/radqm9_pipeline/modules/elements.pkl')

h5_spi_dir='/pscratch/sd/m/mavaylon/new_pipe/sp_raw_data.h5'

import h5py
merged_file = h5py.File(h5_spi_dir, 'r')

merged_data=[]
for point in tqdm(merged_file['raw_data']['raw']):
    point = ast.literal_eval(point.decode('utf-8'))
    merged_data.append(point)

    
def charge_spin_tag(data: list):
    """
    
    """
    
    for item in tqdm(data):
        item['charge_spin'] = str(item['charge'])+'_'+str(item['spin'])
        
charge_spin_tag(merged_data)

def type_tagger(data: list):
    for item in tqdm(data):
        if item['charge_spin'] == item['optimized_parent_charge_spin']:
            item['sp_config_type'] = 'optimized'
        else:
            item['sp_config_type'] = 'vertical'

type_tagger(merged_data)
            
def build_graph(species, position):
    atoms = ase.atoms.Atoms(symbols=species,
                            positions=position)
    mol = AseAtomsAdaptor.get_molecule(atoms)
    graph = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
    return graph

def sp_filter_broken_graphs(data: list):
    broken = []
    good = []
    
    for item in tqdm(data):
        if item['charge_spin'] == '0,1':
            continue
        else:
            graph = build_graph(item['species'], item['geometry'])
            connected = nx.is_connected(graph.graph.to_undirected())
            if not connected:
                broken.append(item)
            else:
                good.append(item)

    return good, broken

filtered_good, broken = sp_filter_broken_graphs(merged_data)

import h5py
with h5py.File('/pscratch/sd/m/mavaylon/new_pipe/sp_broken.h5', 'w') as file:
    g1 =file.create_group('broken')
    bbd=[]
    for item in broken:
        bbd.append(str(item))
    g1.create_dataset('broken_data',data=bbd)

file.close()
import h5py
with h5py.File('/pscratch/sd/m/mavaylon/new_pipe/sp_filtered_out_broken.h5', 'w') as file:
    g1 =file.create_group('good')
    bbd=[]
    for item in filtered_good:
        bbd.append(str(item))
    g1.create_dataset('good_data',data=bbd)