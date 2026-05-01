cd /rds/user/wz369/hpc-work/PINN_dynamics;

# python="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"
python="/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python"
# Experiment 1: PCA 30D

# $python scripts/MIOFlow/01_train.py \
# 	--data_path data/klein_addpop.h5ad \
# 	--config pca --exp_name klein_addpop_pca30 --seed 42
	
# $python scripts/MIOFlow/01_train.py \
# 	--data_path data/klein_addpop.h5ad \
# 	--config pca --exp_name klein_addpop_pca30_run2 --seed 10

# $python scripts/MIOFlow/01_train.py \
# 	--data_path data/klein_addpop.h5ad \
# 	--config pca --exp_name klein_addpop_pca30_run3 --seed 2 &
# wait

# Experiment 2: DM 10

# $python scripts/MIOFlow/01_train.py \
# 	--data_path data/klein_addpop.h5ad \
# 	--obsm_key DM_EigenVectors --n_dims 10 \
# 	--config dm --exp_name klein_addpop_dm10 --seed 10 &


# $python scripts/MIOFlow/01_train.py \
# 	--data_path data/klein_addpop.h5ad \
# 	--obsm_key DM_EigenVectors --n_dims 10 \
# 	--config dm --exp_name klein_addpop_dm10_run2 --seed 2 &


# $python scripts/MIOFlow/01_train.py \
# 	--data_path data/klein_addpop.h5ad \
# 	--obsm_key DM_EigenVectors --n_dims 10 \
# 	--config dm --exp_name klein_addpop_dm10_run3 --seed 42


## Experiment 3:PCA with GAE 

# $python scripts/MIOFlow/01_train_with_gae.py \
#     --data_path data/klein_addpop.h5ad \
#     --obsm_key X_pca --n_dims 30 \
#     --config pca_gae --exp_name klein_addpop_pca30_gae --seed 10 &

# $python scripts/MIOFlow/01_train_with_gae.py \
#     --data_path data/klein_addpop.h5ad \
#     --obsm_key X_pca --n_dims 30 \
#     --config pca_gae \
# 	--exp_name klein_addpop_pca30_gae_run2 --seed 2 &

# $python scripts/MIOFlow/01_train_with_gae.py \
#     --data_path data/klein_addpop.h5ad \
#     --obsm_key X_pca --n_dims 30 \
#     --config pca_gae --exp_name klein_addpop_pca30_gae_run3 --seed 42 &


$python scripts/MIOFlow/01_train_with_gae.py \
	--gaga_latent_dim 15 \
    --data_path data/klein_addpop.h5ad \
    --obsm_key X_pca \
    --obs_time_key Time_point \
    --exp_name klein_pca_gaga_latent15_run1 \
    --seed 42 &
    
$python scripts/MIOFlow/01_train_with_gae.py \
    --gaga_latent_dim 15 \
    --data_path data/klein_addpop.h5ad \
    --obsm_key X_pca \
    --obs_time_key Time_point \
    --exp_name klein_pca_gaga_latent15_run2 \
    --seed 0 &
    
$python scripts/MIOFlow/01_train_with_gae.py \
    --gaga_latent_dim 15 \
    --data_path data/klein_addpop.h5ad \
    --obsm_key X_pca \
    --obs_time_key Time_point \
    --exp_name klein_pca_gaga_latent15_run3 \
    --seed 2 &
wait


## Experiment 4: DM with GAE

# $python scripts/MIOFlow/01_train_with_gae.py \
# 	--data_path data/klein_addpop.h5ad \
# 	--obsm_key DM_EigenVectors --n_dims 10 \
# 	--config dm_gae --exp_name klein_addpop_dm10_gae_run1 --seed 10 &

# $python scripts/MIOFlow/01_train_with_gae.py \
# 	--data_path data/klein_addpop.h5ad \
# 	--obsm_key DM_EigenVectors --n_dims 10 \
# 	--config dm_gae --exp_name klein_addpop_dm10_gae_run2 --seed 2 &

# $python scripts/MIOFlow/01_train_with_gae.py \
# 	--data_path data/klein_addpop.h5ad \
# 	--obsm_key DM_EigenVectors --n_dims 10 \
# 	--config dm_gae --exp_name klein_addpop_dm10_gae_run3 --seed 42 &

# wait