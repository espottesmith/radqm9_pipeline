#!/bin/bash
#SBATCH -A m4298
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128

export SLURM_CPU_BIND="cores"
srun python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/03/rad_qm9_7_20_24_converted_E_F_convrespdm_train_03.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/03/rad_qm9_7_20_24_converted_E_F_convrespdm_val_03.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/03/rad_qm9_7_20_24_converted_E_F_convrespdm_test_03.xyz" \
    --num_process=64 \
    --energy_key='energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/03/h5/" \
    --seed=123 \
    --E0s="average" 
    
    
srun python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/-12/rad_qm9_7_20_24_converted_E_F_convrespdm_train_-12.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/-12/rad_qm9_7_20_24_converted_E_F_convrespdm_val_-12.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/-12/rad_qm9_7_20_24_converted_E_F_convrespdm_test_-12.xyz" \
    --num_process=64 \
    --energy_key='energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/-12/h5/" \
    --seed=123 \
    --E0s="average" 
    
srun python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/-21/rad_qm9_7_20_24_converted_E_F_convrespdm_train_-21.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/-21/rad_qm9_7_20_24_converted_E_F_convrespdm_val_-21.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/-21/rad_qm9_7_20_24_converted_E_F_convrespdm_test_-21.xyz" \
    --num_process=64 \
    --energy_key='energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/-21/h5/" \
    --seed=123 \
    --E0s="average" 
    
srun python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/12/rad_qm9_7_20_24_converted_E_F_convrespdm_train_12.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/12/rad_qm9_7_20_24_converted_E_F_convrespdm_val_12.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/12/rad_qm9_7_20_24_converted_E_F_convrespdm_test_12.xyz" \
    --num_process=64 \
    --energy_key='energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/12/h5/" \
    --seed=123 \
    --E0s="average" 
    
srun python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/21/rad_qm9_7_20_24_converted_E_F_convrespdm_train_21.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/21/rad_qm9_7_20_24_converted_E_F_convrespdm_val_21.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/21/rad_qm9_7_20_24_converted_E_F_convrespdm_test_21.xyz" \
    --num_process=64 \
    --energy_key='energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/smd/21/h5/" \
    --seed=123 \
    --E0s="average" 
    
python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields/smd/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_SMD_train.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields/smd/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_SMD_val.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields/smd/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_SMD_test.xyz" \
    --num_process=64 \
    --energy_key='total_energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields/smd/h5/" \
    --seed=123 \
    --E0s="average" 
# VAC##########
srun python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/03/rad_qm9_7_20_24_converted_E_F_convrespdm_train_vac_03.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/03/rad_qm9_7_20_24_converted_E_F_convrespdm_val_vac_03.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/03/rad_qm9_7_20_24_converted_E_F_convrespdm_test_vac_03.xyz" \
    --num_process=64 \
    --energy_key='energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/03/h5/" \
    --seed=123 \
    --E0s="average" 
    
srun python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-12/rad_qm9_7_20_24_converted_E_F_convrespdm_train_vac_-12.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-12/rad_qm9_7_20_24_converted_E_F_convrespdm_val_vac_-12.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-12/rad_qm9_7_20_24_converted_E_F_convrespdm_test_vac_-12.xyz" \
    --num_process=64 \
    --energy_key='energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-12/h5/" \
    --seed=123 \
    --E0s="average" 
    
srun python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-21/rad_qm9_7_20_24_converted_E_F_convrespdm_train_vac_-21.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-21/rad_qm9_7_20_24_converted_E_F_convrespdm_val_vac_-21.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-21/rad_qm9_7_20_24_converted_E_F_convrespdm_test_vac_-21.xyz" \
    --num_process=64 \
    --energy_key='energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-21/h5/" \
    --seed=123 \
    --E0s="average" 

srun python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/01/rad_qm9_7_20_24_converted_E_F_convrespdm_train_vac_01.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/01/rad_qm9_7_20_24_converted_E_F_convrespdm_val_vac_01.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/01/rad_qm9_7_20_24_converted_E_F_convrespdm_test_vac_01.xyz" \
    --num_process=64 \
    --energy_key='energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/01/h5/" \
    --seed=123 \
    --E0s="average" 

srun python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/21/rad_qm9_7_20_24_converted_E_F_convrespdm_train_vac_21.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/21/rad_qm9_7_20_24_converted_E_F_convrespdm_val_vac_21.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/21/rad_qm9_7_20_24_converted_E_F_convrespdm_test_vac_21.xyz" \
    --num_process=64 \
    --energy_key='energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/21/h5/" \
    --seed=123 \
    --E0s="average" 
    
srun python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/12/rad_qm9_7_20_24_converted_E_F_convrespdm_train_vac_12.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/12/rad_qm9_7_20_24_converted_E_F_convrespdm_val_vac_12.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/12/rad_qm9_7_20_24_converted_E_F_convrespdm_test_vac_12.xyz" \
    --num_process=64 \
    --energy_key='energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/12/h5/" \
    --seed=123 \
    --E0s="average" 
    
python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields/vac/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_VAC_train.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields/vac/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_VAC_val.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields/vac/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_VAC_test.xyz" \
    --num_process=64 \
    --energy_key='total_energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields/vac/h5/" \
    --seed=123 \
    --E0s="average" 
    
python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_Doublet/smd/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_SMD_Doublet_train.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_Doublet/smd/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_SMD_Doublet_val.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_Doublet/smd/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_SMD_Doublet_test.xyz" \
    --num_process=64 \
    --energy_key='total_energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_Doublet/smd/h5/" \
    --seed=123 \
    --E0s="average"

python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_Doublet/vac/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_VAC_Doublet_train.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_Doublet/vac/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_VAC_Doublet_val.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_Doublet/vac/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_VAC_Doublet_test.xyz" \
    --num_process=64 \
    --energy_key='total_energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_Doublet/vac/h5/" \
    --seed=123 \
    --E0s="average"
    
python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_singlet/smd/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_SMD_singlet_train.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_singlet/smd/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_SMD_singlet_val.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_singlet/smd/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_SMD_singlet_test.xyz" \
    --num_process=64 \
    --energy_key='total_energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_singlet/smd/h5/" \
    --seed=123 \
    --E0s="average"

python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_singlet/vac/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_VAC_singlet_train.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_singlet/vac/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_VAC_singlet_val.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_singlet/vac/rad_qm9_7_20_24_converted_E_F_convrespdm_relenergy_VAC_singlet_test.xyz" \
    --num_process=64 \
    --energy_key='total_energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_singlet/vac/h5/" \
    --seed=123 \
    --E0s="average"

########################
python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Zip_FINAL/FullFields_Doublet_chunks/vac/75/rad_qm9_vac_train75.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Zip_FINAL/Full_fields_Singlet_chunks/vac/05/rad_qm9_vac_train05.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Zip_FINAL/Full_fields_Singlet_chunks/vac/05/rad_qm9_vac_train05.xyz" \
    --num_process=64 \
    --energy_key='total_energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Zip_FINAL/FullFields_Doublet_chunks/vac/75/h5/" \
    --seed=123 \
    --E0s="average"

python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-21/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_train_-21.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-21/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_val_-21.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-21/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_test_-21.xyz" \
    --num_process=64 \
    --energy_key='total_energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-21/h5/" \
    --seed=123 \
    --E0s="average"
    
python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/12/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_train_12.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/12/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_val_12.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/12/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_test_12.xyz" \
    --num_process=64 \
    --energy_key='total_energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/12/h5/" \
    --seed=123 \
    --E0s="average"

python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-12/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_train_-12.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-12/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_val_-12.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-12/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_test_-12.xyz" \
    --num_process=64 \
    --energy_key='total_energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/-12/h5/" \
    --seed=123 \
    --E0s="average"

python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/01/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_train_01.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/01/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_val_01.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/01/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_test_01.xyz" \
    --num_process=64 \
    --energy_key='total_energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/01/h5/" \
    --seed=123 \
    --E0s="average"

python /pscratch/sd/m/mavaylon/kareem_mace/mace/scripts/preprocess_data.py \
    --train_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/03/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_train_03.xyz" \
    --valid_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/03/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_val_03.xyz" \
    --test_file="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/03/rad_qm9_7_25_24_converted_E_F_convrespdm_relenergy_VAC_test_03.xyz" \
    --num_process=64 \
    --energy_key='total_energy'\
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --total_charges="[-2, -1, 0, 1, 2]" \
    --spins="[1, 2, 3]" \
    --r_max=5.0 \
    --h5_prefix="/pscratch/sd/m/mavaylon/chem_final_data/SP/Full_Fields_ChargeSPin/vac/03/h5/" \
    --seed=123 \
    --E0s="average"