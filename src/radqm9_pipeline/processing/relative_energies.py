# This file loads an xyz dataset and prepares
# new hdf5 file that is ready for training with on-the-fly dataloading

import logging
import numpy as np
from tqdm import tqdm
import os

import ase

from monty.serialization import dumpfn

from mace import tools
from mace.tools.scripts_utils import get_dataset_from_xyz, get_atomic_energies


def get_all_xyz_files(path: str):
    xyz_files = list()

    for item in os.listdir(path):
        if item.endswith(".xyz"):
            xyz_files.append(os.path.join(path, item))
        
        elif os.path.isdir(os.path.join(path, item)):
            xyz_files += get_all_xyz_files(os.path.join(path, item))

    return xyz_files


def relative_energies(atom_data: list, atomic_energies: dict):
    for item in tqdm(atom_data):
        key = f"{int(item.info['charge'])}_{int(item.info['spin'])}"
        
        lookup_sum = 0
        for num in item.arrays['numbers']:
            lookup_sum += atomic_energies[key][num]
        
        rel = item.info['energy'] - lookup_sum
        item.info['relative_energy'] = rel


def main():
    """
    Load an XYZ-based dataset and calculate the effective average energies of each atom, split by charge-spin state.
    Then, add relative energies to the dataset based on those atomic energies.
    """

    traj_path = ""
    sp_path = ""

    traj_minimal_path = os.path.join(traj_path, "minimal")
    traj_full_path = os.path.join(traj_path, "full")

    sp_vacuum_path = os.path.join(sp_path, "vacuum")
    sp_vacuum_minimal_path = os.path.join(sp_vacuum_path, "minimal")
    sp_vacuum_full_path = os.path.join(sp_vacuum_path, "full")

    sp_solvent_path = os.path.join(sp_path, "smd")
    sp_solvent_minimal_path = os.path.join(sp_solvent_path, "minimal")
    sp_solvent_full_path = os.path.join(sp_solvent_path, "full")

    traj_charge_spins = ["0_1", "0_3", "1_2", "-1_2"]
    sp_charge_spins = ["0_1", "0_3", "1_2", "-1_2", "-2_1", "2_1"]

    atomic_numbers = [1, 6, 7, 8, 9]
    z_table = tools.get_atomic_number_table_from_zs(atomic_numbers)

    for path, charge_spins in [
        (traj_minimal_path, traj_charge_spins),
        (traj_full_path, traj_charge_spins),
        (sp_vacuum_minimal_path, sp_charge_spins),
        (sp_vacuum_full_path, sp_charge_spins),
        (sp_solvent_minimal_path, sp_charge_spins),
        (sp_solvent_full_path, sp_charge_spins)
    ]:
        bychargespin_dir = os.path.join(path, "by_charge_spin")

        all_atomic_energies = dict()

        files = os.listdir(bychargespin_dir)

        for charge_spin in charge_spins:
            train_file = [x for x in files if x.endswith(f"train_{charge_spin}.xyz")][0]
            val_file = [x for x in files if x.endswith(f"val_{charge_spin}.xyz")][0]
            test_file = [x for x in files if x.endswith(f"test_{charge_spin}.xyz")][0]

            collections, atomic_energies_dict = get_dataset_from_xyz(
                train_path=os.path.join(bychargespin_dir, train_file),
                valid_path=os.path.join(bychargespin_dir, val_file),
                test_path=os.path.join(bychargespin_dir, test_file),
                valid_fraction=0.1,
                config_type_weights={"Default": 1.0},
                seed=42,
                energy_key="energy",
                forces_key="forces",
                stress_key="stress",
                virials_key="virials",
                dipole_key="dipole",
                charges_key="charges",
            )

            atomic_energies_dict = get_atomic_energies("average", collections.train, z_table)
            atomic_energies: np.ndarray = np.array(
                [atomic_energies_dict[z] for z in z_table.zs]
            )

            logging.info(f"Atomic energies {path} {charge_spin}: {atomic_energies.tolist()}")
            dumpfn(atomic_energies_dict, os.path.join(path, f"atomic_energies_{charge_spin}.json"))

            all_atomic_energies[charge_spin] = atomic_energies_dict

        for xyz_path in get_all_xyz_files(path):
            atoms_dataset = ase.io.read(xyz_path, index=":")

            relative_energies(atoms_dataset, all_atomic_energies)

            ase.io.write(xyz_path, atoms_dataset, format="extxyz")


if __name__ == "__main__":
    main()
