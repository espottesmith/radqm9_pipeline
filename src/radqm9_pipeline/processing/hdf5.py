# This file loads an xyz dataset and prepares
# new hdf5 file that is ready for training with on-the-fly dataloading

import argparse
import dataclasses
import logging
import ast
import numpy as np
import random
import tqdm
from glob import glob
import h5py
import multiprocessing as mp
import os
from typing import Callable, Dict, List, Optional, Tuple

import ase

from monty.serialization import dumpfn

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
    write_value
)
from mace.tools.scripts_utils import get_atomic_energies
from mace.tools.utils import AtomicNumberTable
from mace.tools import torch_geometric
from mace.modules import compute_statistics


# Note: in trajectory data, dipole moment keys are "dipole_moments", "resp_dipole_moments", and "calc_resp_dipole_moments"
# In sp data, they are "dipole_moment", "resp_dipole_moment", and "calc_resp_dipole_moment". Whoops...


def build_preprocess_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        help="Training set xyz file",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--valid_file",
        help="Validation set xyz file",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--test_file",
        help="Test set xyz file",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--ood_file",
        help="OOD test set xyz file",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--num_process",
        help="The user defined number of processes to use, as well as the number of files created.", 
        type=int, 
        default=int(os.cpu_count()/4)
    )
    parser.add_argument(
        "--valid_fraction",
        help="Fraction of training set used for validation",
        type=float,
        default=0.1,
        required=False,
    )
    parser.add_argument(
        "--h5_prefix",
        help="Prefix for h5 files when saving",
        type=str,
        default="",
    )
    parser.add_argument(
        "--r_max", help="distance cutoff (in Ang)", 
        type=float, 
        default=5.0
    )
    parser.add_argument(
        "--config_type_weights",
        help="String of dictionary containing the weights for each config type",
        type=str,
        default='{"Default":1.0}',
    )
    parser.add_argument(
        "--energy_key",
        help="Key of reference energies in training xyz",
        type=str,
        default="energy",
    )
    parser.add_argument(
        "--forces_key",
        help="Key of reference forces in training xyz",
        type=str,
        default="forces",
    )
    parser.add_argument(
        "--virials_key",
        help="Key of reference virials in training xyz",
        type=str,
        default="virials",
    )
    parser.add_argument(
        "--stress_key",
        help="Key of reference stress in training xyz",
        type=str,
        default="stress",
    )
    parser.add_argument(
        "--dipole_key",
        help="Key of reference dipoles in training xyz",
        type=str,
        default="dipole",
    )
    parser.add_argument(
        "--charges_key",
        help="Key of atomic charges in training xyz",
        type=str,
        default="charges",
    )
    parser.add_argument(
        "--total_charge_key",
        help="Key of molecular charge in training xyz",
        type=str,
        default="charge",
    )
    parser.add_argument(
        "--spin_key",
        help="Key of molecular spin in training xyz",
        type=str,
        default="spin",
    )
    parser.add_argument(
        "--atomic_numbers",
        help="List of atomic numbers",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--total_charges",
        help="List total molecular charges",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--spins",
        help="List of molecular spins",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--compute_statistics",
        help="Compute statistics for the dataset",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--batch_size", 
        help="batch size to compute average number of neighbours", 
        type=int, 
        default=16,
    )
    parser.add_argument(
        "--scaling",
        help="type of scaling to the output",
        type=str,
        default="rms_forces_scaling",
        choices=["std_scaling", "rms_forces_scaling", "no_scaling"],
    )
    parser.add_argument(
        "--E0s",
        help="Dictionary of isolated atom energies",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--shuffle",
        help="Shuffle the training dataset",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--seed",
        help="Random seed for splitting training and validation sets",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--extended",
        help="Store additional data, including various types of calculated atomic partial charges, atomic partial"
             "spins, and dipole moments",
        action="store_true",
        default=False,
    )
    return parser


@dataclasses.dataclass
class ExpandedConfiguration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    relative_energy: Optional[float] = None  # eV
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
    calc_resp_dipole: Optional[Vector] = None  # Debye

    total_charge: Optional[int] = None # molecular charge
    spin: Optional[int] = None # molecular spin
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

    weight: float = 1.0  # weight of config in loss
    energy_weight: float = 1.0  # weight of config energy in loss
    forces_weight: float = 1.0  # weight of config forces in loss
    stress_weight: float = 1.0  # weight of config stress in loss
    virials_weight: float = 1.0  # weight of config virial in loss
    charges_weight: float = 1.0  # weight of config partial charges in loss
    total_charge_weight: float = 1.0  # weight of config total charge in loss
    spins_weight: float = 1.0  # weight of config partial spins in loss
    total_spin_weight: float = 1.0  # weight of config spin multiplicity in loss
    config_type: Optional[str] = "Default"  # config_type of config


ExpandedConfigurations = List[ExpandedConfiguration]


@dataclasses.dataclass
class ExpandedSubsetCollection:
    train: ExpandedConfigurations
    valid: ExpandedConfigurations
    tests: List[Tuple[str, ExpandedConfigurations]]
    ood: ExpandedConfigurations


@dataclasses.dataclass
class SubsetCollection:
    train: data.Configurations
    valid: data.Configurations
    tests: List[Tuple[str, data.Configurations]]
    ood: data.Configurations


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
) -> ExpandedConfigurations:
    """Convert list of ase.Atoms into ExpandedConfigurations"""
    if config_type_weights is None:
        config_type_weights = {"Default": 1.0}

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
    dipole_key="dipole_moments",
    charges_key="charges",
    total_charge_key="mulliken_partial_charges",
    spin_key="spin",
    config_type_weights: Dict[str, float] = None,
) -> ExpandedConfiguration:
    """Convert ase.Atoms to ExpandedConfiguration"""
    if config_type_weights is None:
        config_type_weights = {"Default": 1.0}

    energy = atoms.info.get(energy_key, None)  # eV
    relative_energy = atoms.info.get("relative_energy", None)  # eV
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    stress = atoms.info.get(stress_key, None)  # eV / Ang
    virials = atoms.info.get(virials_key, None)

    # Dipole moments
    dipole = atoms.info.get(dipole_key, None)  # Debye
    if dipole is None and dipole_key == "dipole_moments":
        dipole = atoms.info.get("dipole_moment", None)

    resp_dipole = atoms.info.get("resp_dipole_moments", None)
    if resp_dipole is None:
        resp_dipole = atoms.info.get("resp_dipole_moment", None)

    calc_resp_dipole = atoms.info.get("calc_resp_dipole_moments", None)
    if calc_resp_dipole is None:
        calc_resp_dipole = atoms.info.get("calc_resp_dipole_moment", None)
    
    # Atomic partial charges and spins
    # All charges and spins given in atomic units
    # Charges default to 0 instead of None if not found
    charges = atoms.arrays.get(charges_key, np.zeros(len(atoms)))

    mulliken_charges = atoms.arrays.get("mulliken_partial_charges", np.zeros(len(atoms)))
    mulliken_spins = atoms.arrays.get("mulliken_partial_spins", np.zeros(len(atoms)))
    resp_charges = atoms.arrays.get("resp_partial_charges", np.zeros(len(atoms)))
    nbo_charges = atoms.arrays.get("nbo_partial_charges", np.zeros(len(atoms)))
    nbo_spins = atoms.arrays.get("nbo_partial_spins", np.zeros(len(atoms)))
    
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
    charges_weight = atoms.info.get("config_charges_weight", 1.0)
    total_charge_weight = atoms.info.get("config_total_charge_weight", 1.0)
    spins_weight = atoms.info.get("config_spins_weight", 1.0)
    total_spin_weight = atoms.info.get("config_total_spin_weight", 1.0)

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
    if charges is None:
        charges = np.zeros(len(atomic_numbers))
        charges_weight = 0.0
        total_charge_weight = 0.0
    if mulliken_spins is None and nbo_spins is None:
        mulliken_spins = np.zeros(len(atomic_numbers))
        nbo_spins = np.zeros(len(atomic_numbers))
        spins_weight = 0.0
        total_spin_weight = 0.0

    return ExpandedConfiguration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        energy=energy,
        relative_energy=relative_energy,
        forces=forces,
        stress=stress,
        virials=virials,
        dipole=dipole,
        resp_dipole=resp_dipole,
        calc_resp_dipole=calc_resp_dipole,
        charges=charges,
        mulliken_partial_charges=mulliken_charges,
        mulliken_partial_spins=mulliken_spins,
        resp_partial_charges=resp_charges,
        nbo_partial_charges=nbo_charges,
        nbo_partial_spins=nbo_spins,
        total_charge=total_charge,
        spin=spin,
        weight=weight,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        charges_weight=charges_weight,
        total_charge_weight=total_charge_weight,
        spins_weight=spins_weight,
        total_spin_weight=total_spin_weight,
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
    ood_path: Optional[str] = None,
    seed: int = 1234,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole_moments",
    charges_key: str = "mulliken_partial_charges",
    total_charge_key: str = "charge",
    spin_key: str = "spin",
) -> ExpandedSubsetCollection:
    
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

    ood_configs = []
    if ood_path is not None:
        ood_configs = load_from_xyz_expanded(
            file_path=ood_path,
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
            f"Loaded {len(ood_configs)} OOD test configurations from '{ood_path}'"
        )
    
    return ExpandedSubsetCollection(train=train_configs, valid=valid_configs, tests=test_configs, ood=ood_configs)


def save_expanded_configurations_as_HDF5(configurations: ExpandedConfigurations, i, h5_file) -> None:
    grp = h5_file.create_group("config_batch_0")
    for i, config in enumerate(configurations):
        subgroup_name = f"config_{i}"
        subgroup = grp.create_group(subgroup_name)

        subgroup["atomic_numbers"] = write_value(config.atomic_numbers)
        subgroup["positions"] = write_value(config.positions)
        subgroup["energy"] = write_value(config.energy)
        subgroup["relative_energy"] = write_value(config.relative_energy)
        
        subgroup["forces"] = write_value(config.forces)
        subgroup["stress"] = write_value(config.stress)
        subgroup["virials"] = write_value(config.virials)
        
        subgroup["dipole"] = write_value(config.dipole)
        subgroup["resp_dipole"] = write_value(config.resp_dipole)
        subgroup["calc_resp_dipole"] = write_value(config.calc_resp_dipole)
        
        subgroup["charges"] = write_value(config.charges)
        subgroup["mulliken_partial_charges"] = write_value(config.mulliken_partial_charges)
        subgroup["mulliken_partial_spins"] = write_value(config.mulliken_partial_spins)
        subgroup["resp_partial_charges"] = write_value(config.resp_partial_charges)
        subgroup["nbo_partial_charges"] = write_value(config.nbo_partial_charges)
        subgroup["nbo_partial_spins"] = write_value(config.nbo_partial_spins)

        subgroup["total_charge"] = write_value(config.total_charge)
        subgroup["spin"] = write_value(config.spin)
        subgroup["cell"] = write_value(config.cell)
        subgroup["pbc"] = write_value(config.pbc)

        subgroup["weight"] = write_value(config.weight)
        subgroup["energy_weight"] = write_value(config.energy_weight)
        subgroup["forces_weight"] = write_value(config.forces_weight)
        subgroup["stress_weight"] = write_value(config.stress_weight)
        subgroup["virials_weight"] = write_value(config.virials_weight)
        subgroup["charges_weight"] = write_value(config.charges_weight)
        subgroup["total_charge_weight"] = write_value(config.total_charge_weight)
        subgroup["spins_weight"] = write_value(config.spins_weight)
        subgroup["total_spin_weight"] = write_value(config.total_spin_weight)

        subgroup["config_type"] = write_value(config.config_type)


def get_dataset_from_xyz(
    train_path: str,
    valid_path: str,
    valid_fraction: float,
    config_type_weights: Dict,
    test_path: Optional[str] = None,
    ood_path: Optional[str] = None,
    seed: int = 1234,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipoles",
    charges_key: str = "charges",
    total_charge_key: str = "charge",
    spin_key: str = "spin",
) -> SubsetCollection:
    """Load training and test dataset from xyz file"""
    _, all_train_configs = data.load_from_xyz(
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
        extract_atomic_energies=False,
    )
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )
    if valid_path is not None:
        _, valid_configs = data.load_from_xyz(
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
            extract_atomic_energies=False,
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
        _, all_test_configs = data.load_from_xyz(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            total_charge_key=total_charge_key,
            spin_key=spin_key,
            extract_atomic_energies=False,
        )
        # create list of tuples (config_type, list(Atoms))
        test_configs = data.test_config_types(all_test_configs)
        logging.info(
            f"Loaded {len(all_test_configs)} test configurations from '{test_path}'"
        )

    ood_configs = []
    if ood_path is not None:
        ood_configs = data.load_from_xyz(
            file_path=ood_path,
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
            f"Loaded {len(ood_configs)} OOD test configurations from '{ood_path}'"
        )

    return SubsetCollection(train=train_configs, valid=valid_configs, tests=test_configs, ood=ood_configs)


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
    
    re = [
        pool.apply_async(
            compute_stats_target,
            args=(file, z_table, r_max, atomic_energies, batch_size,)
        ) 
        for file in glob(path_to_files+'/*')
    ]
    
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


def multi_hdf5(data: list, process: int, h5_prefix: str, subset: str, drop_last: bool, save_function: Callable, name: Optional[str] = None):
    if name is None:
        name = subset
    
    h5_file = f"{h5_prefix}/{subset}/{name}_{process}.h5"

    with h5py.File(h5_file, "w") as f:
        f.attrs["drop_last"] = drop_last
        save_function(data[process], process, f)


def main():

    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """
    args = build_preprocess_arg_parser().parse_args() 
    
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
    
    if not os.path.exists(args.h5_prefix):
        os.makedirs(args.h5_prefix)

    # Data preparation
    if args.extended:
        collections = get_expanded_dataset_from_xyz(
            train_path=args.train_file,
            valid_path=args.valid_file,
            valid_fraction=args.valid_fraction,
            config_type_weights=config_type_weights,
            test_path=args.test_file,
            ood_path=args.ood_path,
            seed=args.seed,
            energy_key=args.energy_key,
            forces_key=args.forces_key,
            stress_key=args.stress_key,
            virials_key=args.virials_key,
            dipole_key=args.dipole_key,
            charges_key=args.charges_key,
        )
    else:    
        collections = get_dataset_from_xyz(
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
    if not os.path.exists(os.path.join(args.h5_prefix, "train")):
        os.makedirs(os.path.join(args.h5_prefix, "train"))
    if args.shuffle:
        random.shuffle(collections.train)

    # split collections.train into batches and save them to hdf5
    split_train = np.array_split(collections.train, args.num_process)
    drop_last = False
    if len(collections.train) % 2 == 1:
        drop_last = True

    if args.extended:
        save_function = save_expanded_configurations_as_HDF5
    else:
        save_function = save_configurations_as_HDF5

    processes = []
    for i in range(args.num_process):
        p = mp.Process(
            target=multi_hdf5,
            args=[split_train, i, args.h5_prefix, "train", drop_last, save_function, None]
        )
        p.start()
        processes.append(p)
        
    for i in processes:
        i.join()

    # logging.info("Computing statistics")
    # atomic_energies_dict = get_atomic_energies(args.E0s, collections.train, z_table)
    # atomic_energies: np.ndarray = np.array(
    #     [atomic_energies_dict[z] for z in z_table.zs]
    # )

    # logging.info(f"Atomic energies: {atomic_energies.tolist()}")
    # _inputs = [args.h5_prefix + '/train', z_table, args.r_max, atomic_energies, args.batch_size, args.num_process]
    # avg_num_neighbors, mean, std = pool_compute_stats(_inputs)
    # logging.info(f"Average number of neighbors: {avg_num_neighbors}")
    # logging.info(f"Mean: {mean}")
    # logging.info(f"Standard deviation: {std}")

    # save the statistics as a json
    # statistics = {
    #     "atomic_energies": atomic_energies_dict,
    #     "avg_num_neighbors": avg_num_neighbors,
    #     "mean": mean,
    #     "std": std,
    #     "atomic_numbers": z_table.zs,
    #     "r_max": args.r_max,
    # }

    # dumpfn(statistics, args.h5_prefix + "/statistics.json")
    
    logging.info("Preparing validation set")
    if not os.path.exists(os.path.join(args.h5_prefix, "val")):
        os.makedirs(os.path.join(args.h5_prefix, "val"))
    if args.shuffle:
        random.shuffle(collections.valid)
    split_valid = np.array_split(collections.valid, args.num_process) 
    drop_last = False
    if len(collections.valid) % 2 == 1:
        drop_last = True

    processes = []
    for i in range(args.num_process):
        p = mp.Process(
            target=multi_hdf5,
            args=[split_valid, i, args.h5_prefix, "val", drop_last, save_function, None]
        )
        p.start()
        processes.append(p)
        
    for i in processes:
        i.join()

    if args.test_file is not None:
        logging.info("Preparing test sets")
        if not os.path.exists(os.path.join(args.h5_prefix, "test")):
            os.makedirs(os.path.join(args.h5_prefix, "test"))
        for name, subset in collections.tests:
            drop_last = False
            if len(subset) % 2 == 1:
                drop_last = True
            split_test = np.array_split(subset, args.num_process) 

            processes = []
            for i in range(args.num_process):
                p = mp.Process(
                    target=multi_hdf5,
                    args=[split_test, i, args.h5_prefix, "test", drop_last, save_function, name]
                )
                p.start()
                processes.append(p)
                
            for i in processes:
                i.join()

    logging.info("Preparing OOD test set")
    if args.ood_file is not None:
        if not os.path.exists(os.path.join(args.h5_prefix, "ood")):
            os.makedirs(os.path.join(args.h5_prefix, "ood"))

        split_ood = np.array_split(collections.ood, args.num_process) 
        drop_last = False
        if len(collections.ood) % 2 == 1:
            drop_last = True

        processes = []
        for i in range(args.num_process):
            p = mp.Process(
                target=multi_hdf5,
                args=[split_ood, i, args.h5_prefix, "ood", drop_last, save_function, None]
            )
            p.start()
            processes.append(p)
            
        for i in processes:
            i.join()


if __name__ == "__main__":
    main()
    