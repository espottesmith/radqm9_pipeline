#!/bin/bash
#SBATCH -A m4298
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128

export SLURM_CPU_BIND="cores"
srun python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/Traj/Traj_Zip_Final/Final_Chunked_Singlet_Doublet/singlet/75/rad_qm9_traj_subset_train75.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/Traj/Traj_Zip_Final/Final_Chunked_Singlet_Doublet/doublet/05/rad_qm9_traj_subset_train05.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/Traj/Traj_Zip_Final/Final_Chunked_Singlet_Doublet/doublet/05/rad_qm9_traj_subset_train05.xyz" \
    --num_process=64 \
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-1, 0, 1,]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/Traj/Traj_Zip_Final/Final_Chunked_Singlet_Doublet/singlet/75/h5/" \
    --seed=123 \
    --E0s="average" 