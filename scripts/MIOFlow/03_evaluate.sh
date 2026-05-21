# python="singularity exec /rds/user/wz369/hpc-work/containers/flow.sif python"

cd /home/wz369/rds/hpc-work/PINN_dynamics;python=/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python
# PCA 30
# $python scripts/MIOFlow/03_evaluate.py  \
#     --data_path data/klein_addpop.h5ad  \
#     --model_dir results/MIOFlow/klein_addpop_pca30_gae \
#     --obsm_key X_pca \ 
#     --n_dims 30 --device cpu &


# $python scripts/MIOFlow/03_evaluate.py  \
#     --data_path data/klein_addpop.h5ad  \
#     --model_dir results/MIOFlow/klein_addpop_pca30_gae_run2 \
#     --obsm_key X_pca \ 
#     --n_dims 30 --device cpu &

# $python scripts/MIOFlow/03_evaluate.py  \
#     --data_path data/klein_addpop.h5ad  \
#     --model_dir results/MIOFlow/klein_addpop_pca30_gae_run3 \
#     --obsm_key X_pca \ 
#     --n_dims 30 --device cpu &

# Evaluate DM10
$python scripts/MIOFlow/03_evaluate_with_gae.py \
--data_path data/klein_addpop.h5ad \
--model_dir logs/MIOFlow/klein_dm_gaga_run1 \
--obsm_key DM_EigenVectors --n_dims 10 --device cpu &


$python scripts/MIOFlow/03_evaluate_with_gae.py \
--data_path data/klein_addpop.h5ad \
--model_dir logs/MIOFlow/klein_dm_gaga_run2 \
--obsm_key DM_EigenVectors --n_dims 10 --device cpu &


$python scripts/MIOFlow/03_evaluate_with_gae.py \
--data_path data/klein_addpop.h5ad \
--model_dir logs/MIOFlow/klein_dm_gaga_run3 \
--obsm_key DM_EigenVectors --n_dims 10 --device cpu