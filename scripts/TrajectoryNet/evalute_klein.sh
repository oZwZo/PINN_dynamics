cd /rds/user/wz369/hpc-work/PINN_dynamics
python="/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python"

# $python scripts/TrajectoryNet/03_evaluate.py \
#     --data_dir logs/TrajectoryNet/pca30 \
#     --model_dir logs/TrajectoryNet/pca30/model \
#     --data_path data/klein_addpop.h5ad \
#     --device cpu &

# $python scripts/TrajectoryNet/03_evaluate.py \
#     --data_dir logs/TrajectoryNet/pca30 \
#     --model_dir logs/TrajectoryNet/pca30/model_run2 \
#     --data_path data/klein_addpop.h5ad \
#     --device cpu &

# $python scripts/TrajectoryNet/03_evaluate.py \
#     --data_dir logs/TrajectoryNet/pca30 \
#     --model_dir logs/TrajectoryNet/pca30/model_run3 \
#     --data_path data/klein_addpop.h5ad \
#     --device cpu &

# $python scripts/TrajectoryNet/03_evaluate.py \
#     --data_dir logs/TrajectoryNet/pca30 \
#     --model_dir logs/TrajectoryNet/pca30/model_run4 \
#     --data_path data/klein_addpop.h5ad \
#     --device cpu 

$python scripts/TrajectoryNet/03_evaluate.py \
    --data_dir logs/TrajectoryNet/dm10 \
    --model_dir logs/TrajectoryNet/dm10/model \
    --data_path data/klein_addpop.h5ad \
    --device cpu &

$python scripts/TrajectoryNet/03_evaluate.py \
    --data_dir logs/TrajectoryNet/dm10 \
    --model_dir logs/TrajectoryNet/dm10/model_run2 \
    --data_path data/klein_addpop.h5ad \
    --device cpu &

$python scripts/TrajectoryNet/03_evaluate.py \
    --data_dir logs/TrajectoryNet/dm10 \
    --model_dir logs/TrajectoryNet/dm10/model_run3 \
    --data_path data/klein_addpop.h5ad \
    --device cpu


