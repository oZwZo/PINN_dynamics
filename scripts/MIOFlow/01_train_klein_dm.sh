cd /home/wz369/rds/hpc-work/PINN_dynamics;python=/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python

$python scripts/MIOFlow/01_train_with_gae.py \
    --gaga_latent_dim 10 \
    --data_path data/klein_addpop.h5ad \
    --obsm_key DM_EigenVectors \
    --obs_time_key Time_point \
    --exp_name klein_dm_gaga_run1 \
    --seed 42
    
$python scripts/MIOFlow/01_train_with_gae.py \
    --gaga_latent_dim 10 \
    --data_path data/klein_addpop.h5ad \
    --obsm_key DM_EigenVectors \
    --obs_time_key Time_point \
    --exp_name klein_dm_gaga_run2 \
    --seed 0
    
$python scripts/MIOFlow/01_train_with_gae.py \
    --gaga_latent_dim 10 \
    --data_path data/klein_addpop.h5ad \
    --obsm_key DM_EigenVectors \
    --obs_time_key Time_point \
    --exp_name klein_dm_gaga_run3 \
    --seed 2
	  
