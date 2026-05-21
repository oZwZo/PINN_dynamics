cd /rds/user/wz369/hpc-work/PINN_dynamics;
PYTHON="/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python"

## PREPARE DATA

# python scripts/TrajectoryNet/01_prepare_data.py \
#     --data_path data/klein_addpop.h5ad \
#     --output_dir logs/TrajectoryNet --run_name pca30 \
#     --celltype_col Annotation \
#     --obsm_key X_pca --n_dims 30

## TRAIN PCA

# $PYTHON scripts/TrajectoryNet/02_train.py \
# 		--data_path logs/TrajectoryNet/pca30/klein_train.npz \
# 		--embedding_name X_pca \
# 		--max_dim 30 \
# 		--save_dir logs/TrajectoryNet/pca30/model_run1 \
# 		--dims 128-128-128 \
# 		--niters 10000 \
# 		--gpu 0 &

$PYTHON scripts/TrajectoryNet/02_train.py \
		--data_path logs/TrajectoryNet/pca30/klein_train.npz \
		--embedding_name X_pca \
		--max_dim 30 \
		--save_dir logs/TrajectoryNet/pca30/model_run2 \
		--dims 128-128-128 \
		--niters 10000 \
		--gpu 0 &

$PYTHON scripts/TrajectoryNet/02_train.py \
		--data_path logs/TrajectoryNet/pca30/klein_train.npz \
		--embedding_name X_pca \
		--max_dim 30 \
		--save_dir logs/TrajectoryNet/pca30/model_run3 \
		--dims 128-128-128 \
		--niters 10000 \
		--gpu 0 &
wait



## DM 10

# python scripts/TrajectoryNet/01_prepare_data.py \
#     --data_path data/klein_addpop.h5ad \
#     --output_dir logs/TrajectoryNet --run_name dm10 \
#     --celltype_col Annotation \
#     --obsm_key DM_EigenVectors --n_dims 10

# $PYTHON scripts/TrajectoryNet/02_train.py \
#     --data_path logs/TrajectoryNet/dm10/klein_train.npz \
#     --embedding_name DM_EigenVectors --max_dim 10 \
#     --save_dir logs/TrajectoryNet/dm10/model \
#     --dims 64-64-64 --niters 10000 --gpu 0 &


# $PYTHON scripts/TrajectoryNet/02_train.py \
#     --data_path logs/TrajectoryNet/dm10/klein_train.npz \
#     --embedding_name DM_EigenVectors --max_dim 10 \
#     --save_dir logs/TrajectoryNet/dm10/model_run2 \
#     --dims 64-64-64 --niters 10000 --gpu 0 &


$PYTHON scripts/TrajectoryNet/02_train.py \
    --data_path logs/TrajectoryNet/dm10/klein_train.npz \
    --embedding_name DM_EigenVectors --max_dim 10 \
    --save_dir logs/TrajectoryNet/dm10/model_run3 \
    --dims 64-64-64 --niters 10000 --gpu 0 &

wait