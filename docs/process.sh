#!/bin/bash
#SBATCH -A m4298
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 5:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128

export SLURM_CPU_BIND="cores"
srun python /pscratch/sd/m/mavaylon/sam_ldrd/radqm9_pipeline/src/radqm9_pipeline/processing/process_dataset.py \
    --file="/pscratch/sd/m/mavaylon/chem_final_data/Traj/Chunked_data/FullData/rad_qm9_traj_train05.xyz" \
    --forces_key="forces" \
    --prefix="vac_05_" \
    --directory="/pscratch/sd/m/mavaylon/chem_final_data/Traj/Chunked_data/FullData/h5/chunk05" \

