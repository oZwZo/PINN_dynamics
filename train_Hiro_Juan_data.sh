
COMMAND="python dudt_train_mellon.py -D Hiro_Juan_fulldata_20250513 -M log_pde_params -K DM_EigenVectors --deltax_key Delta_DM -G 0 --timepoint_idx 7  --channels 64,32 --n_dimension 10 --schedule_lr CyclicLR --time_scale_factor 1 --norm_time min_minus --time_sensitive"

# control
${COMMAND} --lr 3e-4 --deltax_weight 1e-2 --batch_size 200 --tol 1e-4 --D_penalty 1

# lr
${COMMAND} --lr 1e-4 --deltax_weight 1e-2 --batch_size 200 --tol 1e-4 --D_penalty 1

# delta x weight
${COMMAND} --lr 3e-4 --deltax_weight 1e-1 --batch_size 200 --tol 1e-4 --D_penalty 1

# D penalty
${COMMAND} --lr 3e-4 --deltax_weight 1e-2 --batch_size 200 --tol 1e-4 --D_penalty 10