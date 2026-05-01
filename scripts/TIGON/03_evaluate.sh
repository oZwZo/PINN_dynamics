cd /rds/user/wz369/hpc-work/PINN_dynamics;
# PY="singularity exec /rds/user/wz369/hpc-work/containers/flow.sif python"
DATA=/rds/user/wz369/hpc-work/PINN_dynamics/data/klein_addpop.h5ad
PY=/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python

# $PY scripts/TIGON/03_evaluate.py \
#     --data_dir logs/TIGON/pca \
#     --checkpoint logs/TIGON/pca/model/ckpt.pth \
#     --data_path data/klein_addpop.h5ad \
#     --device cuda:0

# $PY scripts/TIGON/03_evaluate.py \
#     --data_dir logs/TIGON/pca \
#     --checkpoint logs/TIGON/pca/model_seed2/ckpt_best_itr400.pth \
#     --output_dir logs/TIGON/pca/model_seed2/\
#     --data_path data/klein_addpop.h5ad \
#     --device cuda:1 &

# $PY scripts/TIGON/03_evaluate.py \
#     --data_dir logs/TIGON/pca \
#     --checkpoint logs/TIGON/pca/model_seed3/ckpt_best_itr400.pth \
#     --output_dir logs/TIGON/pca/model_seed3/ \
#     --data_path data/klein_addpop.h5ad \
#     --device cuda:1
        
$PY scripts/TIGON/03_evaluate.py \
    --data_dir logs/TIGON/dm \
    --checkpoint logs/TIGON/dm/model/ckpt.pth \
    --output_dir logs/TIGON/dm/model/ \
    --data_path data/klein_addpop.h5ad \
    --device cpu

$PY scripts/TIGON/03_evaluate.py \
    --data_dir logs/TIGON/dm \
    --checkpoint logs/TIGON/dm/model_seed2/ckpt.pth \
    --output_dir logs/TIGON/dm/model_seed2/ \
    --data_path data/klein_addpop.h5ad \
    --device cpu
   
   
$PY scripts/TIGON/03_evaluate.py \
    --data_dir logs/TIGON/dm \
    --checkpoint logs/TIGON/dm/model_seed3/ckpt.pth \
    --output_dir logs/TIGON/dm/model_seed3/ \
    --data_path data/klein_addpop.h5ad \
    --device cpu
    