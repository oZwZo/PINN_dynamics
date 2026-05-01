python="singularity exec /rds/user/wz369/hpc-work/containers/flow.sif python"

# PCA
# $python scripts/prescient/03_evaluate.py \
#     --model_dir results/PRESCIENT/pca_run/growth_weights-softplus_1_500-1e-06/seed_0 \
#     --data_path data/klein_addpop.h5ad \
#     --obsm_key X_pca --n_dims 30 --device cpu &
    
# $python scripts/prescient/03_evaluate.py   \
#     --data_path data/klein_addpop.h5ad   \
#     --obsm_key X_pca --n_dims 30 \
#     --device cpu  \
#     --model_dir results/PRESCIENT/pca_run/growth_weights-softplus_1_500-1e-06/seed_2 &

# $python scripts/prescient/03_evaluate.py   \
# 	--data_path data/klein_addpop.h5ad   \
#     --obsm_key X_pca --n_dims 30 \
#     --device cpu  \
#     --model_dir results/PRESCIENT/pca_run/growth_weights-softplus_1_500-1e-06/seed_42 &

# wait 

# DM
$python scripts/prescient/03_evaluate.py   \
	--data_path data/klein_addpop.h5ad   \
    --obsm_key DM_EigenVectors --n_dims 10 \
    --device cpu  \
    --model_dir results/PRESCIENT/dm_run/growth_weights-softplus_4_64-1e-06/seed_0 
 
#  $python scripts/prescient/03_evaluate.py   \
# 	--data_path data/klein_addpop.h5ad   \
#     --obsm_key DM_EigenVectors --n_dims 10 \
#     --device cpu  \
#     --model_dir results/PRESCIENT/dm_run/growth_weights-softplus_4_64-1e-06/seed_2 
    
#  $python scripts/prescient/03_evaluate.py   \
# 	--data_path data/klein_addpop.h5ad   \
#     --obsm_key DM_EigenVectors --n_dims 10 \
#     --device cpu  \
#     --model_dir results/PRESCIENT/dm_run/growth_weights-softplus_4_64-1e-06/seed_42 

wait