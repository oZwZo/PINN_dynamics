

# train ery_mk Jul-22
GPU=5
sleep 10s
nohup python main_train.py -D ery_mk -K DM_EigenVectors_multiscaled -M pde_params -G ${GPU} --log_name ery_mk_TM1 --batch_size 50 --channels 32,32 --n_dimension 8 --schedule_lr CyclicLR --lr 3e-4 --timepoint_idx \[0,1,2,3,4,6,8\] --deltax_weight 1e-2 --time_scale_factor 5 --weight_intensity 1 --time_sensitive --norm_time True  &
sleep 10s
nohup python main_train.py -D ery_mk -K DM_EigenVectors_multiscaled -M pde_params -G ${GPU} --log_name ery_mk_TM1 --batch_size 50 --channels 32,32 --n_dimension 8 --schedule_lr CyclicLR --lr 3e-4 --timepoint_idx \[0,1,2,3,4,6,8\] --deltax_weight 1e-3 --time_scale_factor 5 --weight_intensity 1 --time_sensitive --norm_time True  &

GPU=4
sleep 10s
nohup python main_train.py -D ery_mk -K DM_EigenVectors_multiscaled -M pde_params -G ${GPU} --log_name ery_mk_TM1 --batch_size 50 --channels 32,32 --n_dimension 8 --schedule_lr CyclicLR --lr 3e-4 --timepoint_idx \[0,1,2,3,4,6,8\] --deltax_weight 1e-2 --time_scale_factor 5 --weight_intensity 3 --time_sensitive --norm_time True  &
sleep 10s
nohup python main_train.py -D ery_mk -K DM_EigenVectors_multiscaled -M pde_params -G ${GPU} --log_name ery_mk_TM1 --batch_size 50 --channels 32,32 --n_dimension 8 --schedule_lr CyclicLR --lr 3e-4 --timepoint_idx \[0,1,2,3,4,6,8\] --deltax_weight 1e-3 --time_scale_factor 5 --weight_intensity 3 --time_sensitive --norm_time True  &
