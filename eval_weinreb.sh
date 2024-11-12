
LOGS_C0="logs/Weinreb_clone0-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_0/checkpoints/epoch=176-total_loss=4.94458437.ckpt
logs/Weinreb_clone0-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_1/checkpoints/epoch=180-total_loss=3.54002976.ckpt"

LOGS_C1="logs/Weinreb_clone1-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_0/checkpoints/epoch=256-total_loss=26.19643402.ckpt
logs/Weinreb_clone1-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_1/checkpoints/epoch=282-total_loss=26.69228935.ckpt
logs/Weinreb_clone1-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_2/checkpoints/epoch=267-total_loss=1.88309681.ckpt"

LOGS_C2="logs/Weinreb_clone2-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_0/checkpoints/epoch=157-total_loss=27.26894379.ckpt
logs/Weinreb_clone2-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_1/checkpoints/epoch=191-total_loss=3.19271231.ckpt
logs/Weinreb_clone2-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_2/checkpoints/epoch=86-total_loss=2.38408899.ckpt"

LOGS_C3="logs/Weinreb_clone3-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_0/checkpoints/epoch=256-total_loss=15.72236156.ckpt
logs/Weinreb_clone3-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_1/checkpoints/epoch=280-total_loss=8.77423382.ckpt
logs/Weinreb_clone3-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_2/checkpoints/epoch=250-total_loss=8.52438831.ckpt"

LOGS_C4="logs/Weinreb_clone4-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_1/checkpoints/epoch=240-total_loss=2.96112132.ckpt
logs/Weinreb_clone4-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_2/checkpoints/epoch=188-total_loss=18.70672798.ckpt
logs/Weinreb_clone4-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_3/checkpoints/epoch=158-total_loss=2.83117151.ckpt
logs/Weinreb_clone4-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_4/checkpoints/epoch=291-total_loss=2.11715341.ckpt"

LOGS_CV1="
logs/Weinreb_clone0-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_1/checkpoints/epoch=180-total_loss=3.54002976.ckpt
logs/Weinreb_clone1-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_1/checkpoints/epoch=282-total_loss=26.69228935.ckpt
logs/Weinreb_clone3-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_1/checkpoints/epoch=280-total_loss=8.77423382.ckpt
logs/Weinreb_clone4-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_1/checkpoints/epoch=240-total_loss=2.96112132.ckpt
"

eval LOGS="$"LOGS_C${CLONE}
for log in $LOGS;do
    python Explore_Notebook/6.simulation/6-9.pde_params_eval_Weinreb.py $log
    done