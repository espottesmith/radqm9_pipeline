def get_molecule_weight(data: list):
    dict_dist = {}
    data_dict = {}
    for item in tqdm(data):
        species_num = []
        species=''.join((sorted(item['species'])))
        
        for element in item['species']:
            species_num.append(elements_dict[element])

        species_sum = sum(species_num)
        try:
            dict_dist[species].append(species_sum)
            # python does a weird thing floats e.g., {126.15499999999993, 126.15499999999994}
            dict_dist[species] = [dict_dist[species][0]]*len(dict_dist[species])
        except KeyError:
            dict_dist[species] = [species_sum]
        
    return dict_dist

def molecule_weight(data: list, weight_dict):
    for item in tqdm(data):
        weight = weight_dict[''.join((sorted(item['species'])))][0]
        item['molecule_mass'] = weight
        
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

def molecule_weight_ase(data: list, weight_dict):
    for item in tqdm(data):
        species=''.join((sorted(item.get_chemical_symbols())))
        weight = weight_dict[species][0]
        item.info['weight'] = weight
        
def weight_to_data_ase(data: list):
    dict_data = {}
    for item in tqdm(data):
        try:
            dict_data[item.info['weight']].append(item)
        except KeyError:
            dict_data[item.info['weight']] = [item]
    return dict_data

def resolve_duplicate_data(data: list):
    filtered_data = []
    
    bucket_mol_id={}
    for item in tqdm(data):
        try:
            bucket_mol_id[item['mol_id']].append(item)
        except KeyError:
            bucket_mol_id[item['mol_id']] = [item]
    
    mol_id_present_config = {}
    for item in tqdm(data):
        opt_parent = item['optimized_parent_charge_spin'][0]+item['optimized_parent_charge_spin'][1]

        item['dup_identifier'] = item['charge_spin']+'_'+item['sp_config_type']+'_'+opt_parent+'_'+item['solvent']
        try:
            mol_id_present_config[item['mol_id']].append(item['charge_spin']+'_'+item['sp_config_type']+'_'+opt_parent+'_'+item['solvent'])
        except KeyError:
            mol_id_present_config[item['mol_id']] = [item['charge_spin']+'_'+item['sp_config_type']+'_'+opt_parent+'_'+item['solvent']]
    
    # get unique set of configs for each key in mol_id_present_config to use as keys to sample from bucket_mol_id
    for mol_id in tqdm(bucket_mol_id):
        pool = list(set(mol_id_present_config[mol_id]))
        for item in pool:
            for point in bucket_mol_id[mol_id]:
                if point['dup_identifier'] == item:
                    filtered_data.append(point)
                    break
                
    return filtered_data


def resolve_parent_charge_spin(data: list):
    for item in tqdm(data):
        item['optimized_parent_charge_spin']= item['optimized_parent_charge_spin'].split('_')
        
def solvent_convert(data:list):
    unresolved = []
    for item in tqdm(data):
        solv  = item['solvent']
        if solv == 'NONE':
            item['solvent'] = 'vacuum'
        elif solv == 'SOLVENT=WATER':
            item['solvent'] = 'SMD'
        else:
            unresolved.append(item)
    return unresolved

def type_tagger(data: list):
    for item in tqdm(data):
        if item['charge_spin'] == item['optimized_parent_charge_spin']:
            item['sp_config_type'] = 'optimized'
        else:
            item['sp_config_type'] = 'vertical'
            
def charge_spin_tag(data: list):
    for item in tqdm(data):
        item['charge_spin'] = str(item['charge'])+'_'+str(item['spin'])
        
        
def sp_convert_energy(data: list):
    for item in tqdm(data):
        energy = item['energy']
        item['energy'] = energy*27.2114
        
def sp_convert_forces(data: list):
    for item in tqdm(data):
        forces = item['gradient']
        atom_arr = []
        for atom in forces:
            comp_arr = []
            for component in atom:
                new_component = component * 51.42208619083232
                comp_arr.append(new_component)
            atom_arr.append(comp_arr)
        item['gradient'] = atom_arr
        
def sp_generate_resp_dipole(data: list): #THIS IS GOOD
    for item in tqdm(data):
        resp_dipole = []
        resp_dipole_conv = []
        
        resp_partial_charges = np.array(item['resp_partial_charges'])
        geometries = np.array(item['geometry'])

        # Calculate dipole moment components
        dipole_components = resp_partial_charges[:, np.newaxis] * geometries

        # Sum the dipole moment components along axis 0 to get the total dipole moment vector
        dipole_moment_conv = np.sum(dipole_components, axis=0)*(1/0.2081943)

        # Append dipole moment to resp_dipole list
        resp_dipole_conv.append(dipole_moment_conv.tolist())  # Convert numpy array to list
        
        item['calc_resp_dipole_moments'] = resp_dipole_conv
        
def build_sp_atoms(data: dict,
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
            positions=data['geometry']
        )
        atoms.arrays['mulliken_partial_charges']=np.array(data['mulliken_partial_charges'])
        atoms.arrays['mulliken_partial_spins']=np.array(data['mulliken_partial_spins'])
        atoms.arrays['resp_partial_charges']=np.array(data['resp_partial_charges'])
        atoms.info['calc_resp_dipole_moments']=np.array(data['calc_resp_dipole_moments'])
        
        atoms.info['optimized_parent_charge']= data['optimized_parent_charge_spin'] [0]
        atoms.info['optimized_parent_spin']= data['optimized_parent_charge_spin'] [0]
        atoms.info['solvent'] =  item.info['solvent'] 

        if energy is not None:
            atoms.info['energy'] = data[energy]
        if forces is not None:
            atoms.arrays['forces'] = np.array(data[forces])
        if charge is not None:
             atoms.info['charge'] = data[charge]
        if spin is not None:
            atoms.info['spin'] = data[spin]
        atoms.info['mol_id'] = data['mol_id']
        
        atom_list.append(atoms)
    return atom_list

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
        
def re_sp_build_minimal_atoms_rel(data: dict,
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
        atoms.info['total_energy'] = item.info['energy']
        atoms.info['relative_energy'] = item.info['relative_energy']
        atoms.info['mol_id'] = item.info['mol_id']
        atoms.arrays['forces'] = np.array(item.arrays['forces'])
        atoms.info['charge'] =  item.info['charge']
        atoms.info['spin'] =  item.info['spin'] 
        atoms.info['optimized_parent_charge'] = item.info['optimized_parent_charge']
        atoms.info['optimized_parent_spin'] = item.info['optimized_parent_spin']
        atoms.info['solvent'] =  item.info['solvent'] 
        atoms.arrays['mulliken_partial_charges']=np.array(item.arrays['mulliken_partial_charges'])
        atoms.arrays['mulliken_partial_spins']=np.array(item.arrays['mulliken_partial_spins'])
        atoms.arrays['resp_partial_charges']=np.array(item.arrays['resp_partial_charges'])
        atoms.info['calc_resp_dipole_moments']=np.array(item.info['calc_resp_dipole_moments'])
        
        atom_list.append(atoms)
    return atom_list
