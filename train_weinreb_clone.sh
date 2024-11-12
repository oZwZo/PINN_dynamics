CLONE=$1
GPU=$2

nohup python dudt_train.py -D Weinreb_clone${CLONE} -K DM_EigenVectors_multiscaled -M pde_params -G ${GPU} --batch_size 100 --channels 6,32,32,1 --n_dimension 5 --schedule_lr CyclicLR --lr 3e-4 --n_timepoint 3 --deltax_key Delta_DM_clone --deltax_weight 1e-2 --time_sensitive &
echo $!

sleep 1s
nohup python dudt_train.py -D Weinreb_clone${CLONE} -K DM_EigenVectors_multiscaled -M pde_params -G ${GPU} --batch_size 100 --channels 6,32,32,1 --n_dimension 5 --schedule_lr CyclicLR --lr 3e-4 --n_timepoint 3 --deltax_key Delta_DM --deltax_weight 1e-2 --time_sensitive &
echo $!

sleep 1s
nohup python dudt_train.py -D Weinreb_clone${CLONE} -K DM_EigenVectors_multiscaled -M pde_params -G ${GPU} --batch_size 100 --channels 6,32,32,1 --n_dimension 5 --schedule_lr CyclicLR --lr 3e-4 --n_timepoint 3 --deltax_key Delta_DM_clone --deltax_weight 1e-2 --time_sensitive --pretrained logs/klein_subset-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_5/checkpoints/epoch=17-total_loss=0.31139943.ckpt &
echo $!