# This file loads an xyz dataset and prepares
# new hdf5 file that is ready for training with on-the-fly dataloading

import logging
import ast
import numpy as np
import json
import random
import tqdm
from glob import glob
import h5py
import multiprocessing as mp
import os
from typing import List, Tuple


from mace import tools, data
from mace.data.utils import (
    Vector,
    Positions,
    Forces,
    Stress,
    Virials,
    Charges,
    Cell,
    Pbc,
    save_configurations_as_HDF5,
)
from mace.tools.scripts_utils import get_dataset_from_xyz, get_atomic_energies
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric
from mace.modules import compute_statistics    


# Note: in trajectory data, dipole moment keys are "dipole_moments", "resp_dipole_moments", and "calc_resp_dipole_moments"
# In sp data, they are "dipole_moment", "resp_dipole_moment", and "calc_resp_dipole_moment". Whoops...


@dataclass
class ExpandedConfiguration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    stress: Optional[Stress] = None  # eV/Angstrom^3
    virials: Optional[Virials] = None  # eV

    # TODO: do we want this? And if we do, what kind of charges go here?
    charges: Optional[Charges] = None  # atomic unit
    mulliken_partial_charges: Optional[Charges] = None  # atomic unit
    mulliken_partial_spins: Optional[Charges] = None  # atomic unit
    resp_partial_charges: Optional[Charges] = None  # atomic unit
    nbo_partial_charges: Optional[Charges] = None  # atomic unit
    nbo_partial_spins: Optional[Charges] = None  # atomic unit

    dipole: Optional[Vector] = None  # Debye
    resp_dipole: Optional[Vector] = None  # Debye
    calc_resp_dipole: Optiona[Vector] = None  # Debye

    total_charge: Optional[int] = None # molecular charge
    spin: Optional[int] = None # molecular spin
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

    weight: float = 1.0  # weight of config in loss
    energy_weight: float = 1.0  # weight of config energy in loss
    forces_weight: float = 1.0  # weight of config forces in loss
    stress_weight: float = 1.0  # weight of config stress in loss
    virials_weight: float = 1.0  # weight of config virial in loss
    config_type: Optional[str] = "Default"  # config_type of config


ExpandedConfigurations = List[ExpandedConfiguration]


@dataclasses.dataclass
class ExpandedSubsetCollection:
    train: ExpandedConfigurations
    valid: ExpandedConfigurations
    tests: List[Tuple[str, ExpandedConfigurations]]


def expanded_config_from_atoms_list(
    atoms_list: List[ase.Atoms],
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole_moments",
    charges_key="mulliken_partial_charges",
    total_charge_key="charge",
    spin_key="spin",
    config_type_weights: Dict[str, float] = None,
) -> Configurations:
    """Convert list of ase.Atoms into Configurations"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    all_configs = []
    for atoms in atoms_list:
        all_configs.append(
            expanded_config_from_atoms(
                atoms,
                energy_key=energy_key,
                forces_key=forces_key,
                stress_key=stress_key,
                virials_key=virials_key,
                dipole_key=dipole_key,
                charges_key=charges_key,
                total_charge_key=total_charge_key,
                spin_key=spin_key,
                config_type_weights=config_type_weights,
            )
        )
    return all_configs


def expanded_config_from_atoms(
    atoms: ase.Atoms,
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    total_charge_key="charge",
    spin_key="spin",
    config_type_weights: Dict[str, float] = None,
) -> Configuration:
    """Convert ase.Atoms to Configuration"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    energy = atoms.info.get(energy_key, None)  # eV
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    stress = atoms.info.get(stress_key, None)  # eV / Ang
    virials = atoms.info.get(virials_key, None)
    dipole = atoms.info.get(dipole_key, None)  # Debye
    # Charges default to 0 instead of None if not found
    charges = atoms.arrays.get(charges_key, np.zeros(len(atoms)))  # atomic unit
    total_charge = atoms.info.get(total_charge_key, 0)
    spin = atoms.info.get(spin_key, 0)
    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    config_type = atoms.info.get("config_type", "Default")
    weight = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
        config_type, 1.0
    )
    energy_weight = atoms.info.get("config_energy_weight", 1.0)
    forces_weight = atoms.info.get("config_forces_weight", 1.0)
    stress_weight = atoms.info.get("config_stress_weight", 1.0)
    virials_weight = atoms.info.get("config_virials_weight", 1.0)

    # fill in missing quantities but set their weight to 0.0
    if energy is None:
        energy = 0.0
        energy_weight = 0.0
    if forces is None:
        forces = np.zeros(np.shape(atoms.positions))
        forces_weight = 0.0
    if stress is None:
        stress = np.zeros(6)
        stress_weight = 0.0
    if virials is None:
        virials = np.zeros((3, 3))
        virials_weight = 0.0
    if dipole is None:
        dipole = np.zeros(3)
        # dipoles_weight = 0.0

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        energy=energy,
        forces=forces,
        stress=stress,
        virials=virials,
        dipole=dipole,
        charges=charges,
        total_charge=total_charge,
        spin=spin,
        weight=weight,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
    )


def load_from_xyz_expanded(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole_moments",
    charges_key: str = "mulliken_partial_charges",
    total_charge_key: str = "charge",
    spin_key: str = "spin",
) -> Tuple[Dict[int, float], ExpandedConfigurations]:

    atoms_list = ase.io.read(file_path, index=":")

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    configs = expanded_config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        total_charge_key=total_charge_key,
        spin_key=spin_key,
    )

    return configs


def get_expanded_dataset_from_xyz(
    train_path: str,
    valid_path: str,
    valid_fraction: float,
    config_type_weights: Dict,
    test_path: Optional[str] = None,
    seed: int = 1234,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole_moments",
    charges_key: str = "mulliken_partial_charges",
    total_charge_key: str = "charge",
    spin_key: str = "spin",
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    
    """Load training, validation, and test datasets from xyz files"""

    all_train_configs = load_from_xyz_expanded(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        total_charge_key=total_charge_key,
        spin_key=spin_key,
    )
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )
    if valid_path is not None:
        valid_configs = load_from_xyz_expanded(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            total_charge_key=total_charge_key,
            spin_key=spin_key,
        )
        logging.info(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    else:
        logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction
        )
        train_configs, valid_configs = data.random_train_valid_split(
            all_train_configs, valid_fraction, seed
        )

    test_configs = []
    if test_path is not None:
        all_test_configs = load_from_xyz_expanded(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            total_charge_key=total_charge_key,
            spin_key=spin_key,
        )
        # create list of tuples (config_type, list(Atoms))
        test_configs = data.test_config_types(all_test_configs)
        logging.info(
            f"Loaded {len(all_test_configs)} test configurations from '{test_path}'"
        )
    
    return ExpandedSubsetCollection(train=train_configs, valid=valid_configs, tests=test_configs)


def compute_stats_target(file: str, z_table: AtomicNumberTable, r_max: float, atomic_energies: Tuple, batch_size: int):
    train_dataset = data.HDF5Dataset(file, z_table=z_table, r_max=r_max)
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False,
    )
    
    avg_num_neighbors, mean, std = compute_statistics(train_loader, atomic_energies)
    output = [avg_num_neighbors, mean, std]
    return output


def pool_compute_stats(inputs: List): 
    path_to_files, z_table, r_max, atomic_energies, batch_size, num_process = inputs
    pool = mp.Pool(processes=num_process)
    
    re=[pool.apply_async(compute_stats_target, args=(file, z_table, r_max, atomic_energies, batch_size,)) for file in glob(path_to_files+'/*')]
    
    pool.close()
    pool.join()
    results = [r.get() for r in tqdm.tqdm(re)]
    return np.average(results, axis=0)


def split_array(a: np.ndarray, max_size: int):
    drop_last = False
    if len(a) % 2 == 1:
        a = np.append(a, a[-1])
        drop_last = True
    factors = get_prime_factors(len(a))
    max_factor = 1
    for i in range(1, len(factors) + 1):
        for j in range(0, len(factors) - i + 1):
            if np.prod(factors[j : j + i]) <= max_size:
                test = np.prod(factors[j : j + i])
                if test > max_factor:
                    max_factor = test
    return np.array_split(a, max_factor), drop_last


def get_prime_factors(n: int):
    factors = []
    for i in range(2, n + 1):
        while n % i == 0:
            factors.append(i)
            n = n / i
    return factors


def main():

    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """
    args = tools.build_preprocess_arg_parser().parse_args() 
    
    # Setup
    tools.set_seeds(args.seed)
    random.seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    
    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}
     
    # Data preparation
    collections, atomic_energies_dict = get_dataset_from_xyz(
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file,
        seed=args.seed,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        virials_key=args.virials_key,
        dipole_key=args.dipole_key,
        charges_key=args.charges_key,
    )

    # Expanded collections - including extra (non-essential) properties
    exp_collections, exp_atomic_energies_dict = get_expanded_dataset_from_xyz(
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file,
        seed=args.seed,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        virials_key=args.virials_key,
        dipole_key=args.dipole_key,
        charges_key=args.charges_key,
    )

    # Atomic number table
    # yapf: disable
    if args.atomic_numbers is None:
        z_table = tools.get_atomic_number_table_from_zs(
            z
            for configs in (collections.train, collections.valid)
            for config in configs
            for z in config.atomic_numbers
        )
    else:
        logging.info("Using atomic numbers from command line argument")
        zs_list = ast.literal_eval(args.atomic_numbers)
        assert isinstance(zs_list, list)
        z_table = tools.get_atomic_number_table_from_zs(zs_list)

    logging.info("Preparing training set")
    if args.shuffle:
        random.shuffle(collections.train)

    # split collections.train into batches and save them to hdf5
    split_train = np.array_split(collections.train,args.num_process)
    drop_last = False
    if len(collections.train) % 2 == 1:
        drop_last = True
    
    # Define Task for Multiprocessiing
    def multi_train_hdf5(process):
        with h5py.File(args.h5_prefix + "train/train_" + str(process)+".h5", "w") as f:
            f.attrs["drop_last"] = drop_last
            save_configurations_as_HDF5(split_train[process], process, f)
      
    processes = []
    for i in range(args.num_process):
        p = mp.Process(target=multi_train_hdf5, args=[i])
        p.start()
        processes.append(p)
        
    for i in processes:
        i.join()


    logging.info("Computing statistics")
    if len(atomic_energies_dict) == 0:
        atomic_energies_dict = get_atomic_energies(args.E0s, collections.train, z_table)
    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )
    logging.info(f"Atomic energies: {atomic_energies.tolist()}")
    _inputs = [args.h5_prefix+'train', z_table, args.r_max, atomic_energies, args.batch_size, args.num_process]
    avg_num_neighbors, mean, std=pool_compute_stats(_inputs)
    logging.info(f"Average number of neighbors: {avg_num_neighbors}")
    logging.info(f"Mean: {mean}")
    logging.info(f"Standard deviation: {std}")

    # save the statistics as a json
    statistics = {
        "atomic_energies": str(atomic_energies_dict),
        "avg_num_neighbors": avg_num_neighbors,
        "mean": mean,
        "std": std,
        "atomic_numbers": str(z_table.zs),
        "r_max": args.r_max,
    }
    
    with open(args.h5_prefix + "statistics.json", "w") as f:
        json.dump(statistics, f)
    
    logging.info("Preparing validation set")
    if args.shuffle:
        random.shuffle(collections.valid)
    split_valid = np.array_split(collections.valid, args.num_process) 
    drop_last = False
    if len(collections.valid) % 2 == 1:
        drop_last = True

    def multi_valid_hdf5(process):
        with h5py.File(args.h5_prefix + "val/val_" + str(process)+".h5", "w") as f:
            f.attrs["drop_last"] = drop_last
            save_configurations_as_HDF5(split_valid[process], process, f)
    
    processes = []
    for i in range(args.num_process):
        p = mp.Process(target=multi_valid_hdf5, args=[i])
        p.start()
        processes.append(p)
        
    for i in processes:
        i.join()

    if args.test_file is not None:
        def multi_test_hdf5(process, name):
            with h5py.File(args.h5_prefix + "test/" + name + "_" + str(process) + ".h5", "w") as f:                    
                f.attrs["drop_last"] = drop_last
                save_configurations_as_HDF5(split_test[process], process, f)
            
        logging.info("Preparing test sets")
        for name, subset in collections.tests:
            drop_last = False
            if len(subset) % 2 == 1:
                drop_last = True
            split_test = np.array_split(subset, args.num_process) 

            processes = []
            for i in range(args.num_process):
                p = mp.Process(target=multi_test_hdf5, args=[i, name])
                p.start()
                processes.append(p)

            for i in processes:
                i.join()

if __name__ == "__main__":
    main()
    