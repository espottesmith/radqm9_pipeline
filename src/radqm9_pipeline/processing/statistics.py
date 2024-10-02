import ast
from glob import glob
import logging
import multiprocessing as mp
from typing import List, Tuple

import tqdm

import numpy as np

import torch_geometric

from mace import tools, data
from mace.tools.utils import AtomicNumberTable
from mace.modules import compute_statistics

from radqm9_pipeline.processing.data_arg_parser import build_default_arg_parser


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


if __name__ == "__main__":
    args = build_default_arg_parser().parse_args()

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
    
    logging.info("Computing statistics")
    logging.info("Ignoring atomic energies")
    atomic_energies_dict = {z: 0.0 for z in z_table.zs}
    atomic_energies: np.ndarray = np.array(
        [0.0 for z in z_table.zs]
    )
    _inputs = [args.prefix, z_table, args.r_max, atomic_energies, args.batch_size, args.num_process]
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
    