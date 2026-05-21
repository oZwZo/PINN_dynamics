PY="singularity exec ../containers/flow.sif python"


# $PY scripts/DeepRUOT/03_evaluate.py \
#     --model_dir /rds/user/wz369/hpc-work/DeepRUOTv2/results/klein_pca30 \
#     --data_path data/klein_addpop.h5ad \
#     --obsm_key X_pca --n_dims 30 --device cpu


# $PY scripts/DeepRUOT/03_evaluate.py \
#     --model_dir /rds/user/wz369/hpc-work/DeepRUOTv2/results/klein_pca30_s0 \
#     --data_path data/klein_addpop.h5ad \
#     --obsm_key X_pca --n_dims 30 --device cpu


# $PY scripts/DeepRUOT/03_evaluate.py \
#     --model_dir /rds/user/wz369/hpc-work/DeepRUOTv2/results/klein_pca30_s10 \
#     --data_path data/klein_addpop.h5ad \
#     --obsm_key X_pca --n_dims 30 --device cpu


# $PY scripts/DeepRUOT/03_evaluate.py \
#     --model_dir /rds/user/wz369/hpc-work/DeepRUOTv2/results/klein_pca30_s42 \
#     --data_path data/klein_addpop.h5ad \
#     --obsm_key X_pca --n_dims 30 --device cpu

$PY scripts/DeepRUOT/03_evaluate.py \
    --model_dir /rds/user/wz369/hpc-work/DeepRUOTv2/results/klein_dm10 \
    --data_path data/klein_addpop.h5ad \
    --obsm_key DM_EigenVectors --n_dims 10 --device cpu

$PY scripts/DeepRUOT/03_evaluate.py \
    --model_dir /rds/user/wz369/hpc-work/DeepRUOTv2/results/klein_dm10_s0 \
    --data_path data/klein_addpop.h5ad \
    --obsm_key DM_EigenVectors --n_dims 10 --device cpu

$PY scripts/DeepRUOT/03_evaluate.py \
    --model_dir /rds/user/wz369/hpc-work/DeepRUOTv2/results/klein_dm10_s10 \
    --data_path data/klein_addpop.h5ad \
    --obsm_key DM_EigenVectors --n_dims 10 --device cpu

$PY scripts/DeepRUOT/03_evaluate.py \
    --model_dir /rds/user/wz369/hpc-work/DeepRUOTv2/results/klein_dm10_s42 \
    --data_path data/klein_addpop.h5ad \
    --obsm_key DM_EigenVectors --n_dims 10 --device cpu