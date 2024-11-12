key=$1
for CLONE in {0..3}; do
    sleep 1s
    nohup python dudt_train.py -D Weinreb_clone${CLONE} -K DM_EigenVectors_multiscaled -M pde_params -G 0 --batch_size 100 --channels 6,32,32,1 --n_dimension 5 --schedule_lr CyclicLR --lr 3e-4 --n_timepoint 3 --deltax_key ${key} --deltax_weight 1e-2 --time_sensitive > Delta_DM_clone.log &
    echo $!
    # sleep 1s
    # nohup python dudt_train.py -D Weinreb_clone${CLONE} -K DM_EigenVectors_multiscaled -M pde_params -G 1 --batch_size 100 --channels 6,32,32,1 --n_dimension 5 --schedule_lr CyclicLR --lr 3e-4 --n_timepoint 3 --deltax_key Delta_DM --deltax_weight 1e-2 --time_sensitive &
    # echo $!

    # sleep 1s
    # nohup python dudt_train.py -D Weinreb_clone${CLONE} -K DM_EigenVectors_multiscaled -M pde_params -G 2 --batch_size 100 --channels 6,32,32,1 --n_dimension 5 --schedule_lr CyclicLR --lr 3e-4 --n_timepoint 3 --deltax_key Delta_DM_clone --deltax_weight 1e-2 --time_sensitive --pretrained logs/klein_subset-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_4/checkpoints/epoch=53-total_loss=0.21702482.ckpt &
    # echo $!

    # sleep 1s
    # nohup python dudt_train.py -D Weinreb_clone${CLONE} -K DM_EigenVectors_multiscaled -M pde_params -G 3 --batch_size 100 --channels 6,32,32,1 --n_dimension 5 --schedule_lr CyclicLR --lr 3e-4 --n_timepoint 3 --deltax_key Delta_DM --deltax_weight 1e-2 --time_sensitive --pretrained logs/klein_subset-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_4/checkpoints/epoch=53-total_loss=0.21702482.ckpt &
    # echo $! 
    done