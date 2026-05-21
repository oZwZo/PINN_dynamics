
LOGNAME="Qi_ALL_pretrain"
COMMAND="python main_train_fastmode.py -D Qi_atlas_ALL_250514 -K DM_EigenVectors --log_name ${LOGNAME} --deltax_key Delta_DM -G 0 --timepoint_key Timepoint --timepoint_idx None --resolution 40  --channels 64,32 --n_dimension 10 --schedule_lr CyclicLR --time_scale_factor 1 --norm_time min_minus --time_sensitive"

# control
${COMMAND} --lr 3e-4 --deltax_weight 1e-2 --batch_size 512 --tol 1e-4 --D_penalty 1 &
sleep 10s;

# # lr
${COMMAND} --lr 3e-3 --deltax_weight 1e-2 --batch_size 512 --tol 1e-4 --D_penalty 1 &
sleep 10s;

# # delta x weight
${COMMAND} --lr 3e-4 --deltax_weight 1    --batch_size 512 --tol 1e-4 --D_penalty 1 &
sleep 10s;
${COMMAND} --lr 3e-4 --deltax_weight 1e-1 --batch_size 512 --tol 1e-4 --D_penalty 1 &
sleep 10s;
${COMMAND} --lr 3e-4 --deltax_weight 1e-3 --batch_size 512 --tol 1e-4 --D_penalty 1 &
sleep 10s;

# # D penalty
${COMMAND} --lr 3e-4 --deltax_weight 1e-2 --batch_size 512 --tol 1e-4 --D_penalty 10
sleep 10s;
