import argparse
import ast
from glob import glob
import logging
import multiprocessing as mp
import os
from typing import List, Tuple

import tqdm

import numpy as np

from mace import tools, data
from mace_tools import torch_geometric
from mace.tools.utils import AtomicNumberTable, TotalChargeTable, SpinTable
from mace.modules import compute_statistics

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--prefix",
        help="The prefix for the h5 file.",
        type=str,
        required = True
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
        help="List of charges",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--spins",
        help="List of spin multiplicities",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--num_process",
        help="The user defined number of processes to use(the number of files created).", 
        type=int, 
        default=int(os.cpu_count()/4)
    )
    parser.add_argument(
        "--E0s",
        help="Dictionary of isolated atom energies",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--r_max",
        help="distance cutoff (in Ang)", 
        type=float, 
        default=5.0
    )
    parser.add_argument(
        "--batch_size", 
        help="batch size", 
        type=int, 
        default=10)
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
   
    return parser



def compute_stats_target(
    file: str,
    z_table: AtomicNumberTable,
    r_max: float,
    atomic_energies: Tuple,
    batch_size: int,
    total_charge_table: TotalChargeTable,
    spin_table: SpinTable):
    train_dataset = data.HDF5Dataset(
        file,
        z_table=z_table,
        r_max=r_max,
        total_charge_table=total_charge_table,
        spin_table=spin_table
    )
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
    path_to_files, z_table, r_max, atomic_energies, batch_size, total_charges, spins, num_process = inputs
    pool = mp.Pool(processes=num_process)
    
    re=[pool.apply_async(compute_stats_target, args=(file, z_table, r_max, atomic_energies, batch_size, total_charges, spins)) for file in glob(path_to_files+'/*')]
    
    pool.close()
    pool.join()
    results = [r.get() for r in tqdm.tqdm(re)]
    return np.average(results, axis=0)


#TODO: TypeError: HDF5Dataset.__init__() missing 2 required positional arguments: 'total_charge_table' and 'spin_table'

if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        # Very 
        config_type_weights = {"Default": 1.0}

    if args.atomic_numbers is None:
        raise ValueError("No atomic numbers provided!")

    # logging.info("Using atomic numbers from command line argument")
    zs_list = ast.literal_eval(args.atomic_numbers)
    assert isinstance(zs_list, list)
    z_table = tools.get_atomic_number_table_from_zs(zs_list)

    charges_list = ast.literal_eval(args.total_charges)
    assert isinstance(charges_list, list)
    assert all([isinstance(x, int) for x in charges_list])
    charges_table = TotalChargeTable(charges_list)

    spins_list = ast.literal_eval(args.spins)
    assert isinstance(spins_list, list)
    assert all([isinstance(x, int) for x in spins_list])
    spins_table = SpinTable(spins_list)
    
    logging.info("Computing statistics")
    logging.info("Ignoring atomic energies")
    atomic_energies_dict = {z: 0.0 for z in z_table.zs}
    atomic_energies: np.ndarray = np.array(
        [0.0 for z in z_table.zs]
    )
    _inputs = [args.prefix, z_table, args.r_max, atomic_energies, args.batch_size, charges_table, spins_table, args.num_process]
    avg_num_neighbors, mean, std = pool_compute_stats(_inputs)
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
    
    with open(args.prefix + "/statistics.json", "w") as f:
        json.dump(statistics, f)
    