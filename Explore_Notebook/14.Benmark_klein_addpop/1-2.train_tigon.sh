PY=/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python
S=/rds/user/wz369/hpc-work/PINN_dynamics/scripts/TIGON
DATA=/rds/user/wz369/hpc-work/PINN_dynamics/data/klein_addpop.h5ad
LOGS=/rds/user/wz369/hpc-work/PINN_dynamics/logs/TIGON
cd /rds/user/wz369/hpc-work/PINN_dynamics

$PY $S/02_train.py \
    --input_dir $LOGS/pca \
    --dataset klein_train \
    --save_dir $LOGS/pca/model_seed2 \
    --hidden_dim 64 --niters 5000 --seed 2 --gpu 0 &


$PY $S/02_train.py \
    --input_dir $LOGS/pca \
    --dataset klein_train \
    --save_dir $LOGS/pca/model_seed3 \
    --hidden_dim 64 --niters 5000 --seed 3 --gpu 0 &


$PY $S/02_train.py \
    --input_dir $LOGS/dm \
    --dataset klein_train \
    --save_dir $LOGS/dm/model_seed2 \
    --hidden_dim 64 --niters 5000 \
    --seed 2 --gpu 1 & 


$PY $S/02_train.py \
    --input_dir $LOGS/dm \
    --dataset klein_train \
    --save_dir $LOGS/dm/model_seed3 \
    --hidden_dim 64 --niters 5000 \
    --seed 3 --gpu 1  &