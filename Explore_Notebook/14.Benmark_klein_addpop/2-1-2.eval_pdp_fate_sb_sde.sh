#!/bin/bash

run_evaluation() {
    local config=$1
    python=/local/scratch/wz369/PINN_env/bin/python
    pdp_dir=/rds/user/wz369/hpc-work/pseudodynamics_plus

    log=${pdp_dir}/${config%.json}_eval_fate_sb_sde.log
    echo "Runing Explore_Notebook/14.Benmark_klein_addpop/2-1-2.eval_pdp_fate_sb_sde.py" >> $log
    echo "Using the score-guided SDE (Schrödinger Bridge) simulation_func to simulate cell states" >> $log

    # for time in 0.2 0.5 1.0 2.0; do
    #     for noise in 1.5 2.0 2.5 3.0; do
    for time in 0.5 4.0 6.0; do
        for noise in 0.5 1.0 1.5; do
            echo "time: $time, noise: $noise" >> $log
            $python Explore_Notebook/14.Benmark_klein_addpop/2-1-2.eval_pdp_fate_sb_sde.py --config_path $pdp_dir/$config --t_end_norm $time --noise_scale $noise >> $log
        done
    done
}

run_evaluation "logs/klein_DM_10_lD10_lv1_lgNone/pde_params_tsense/V0_config.json" &
run_evaluation "logs/klein_DM_10_lD1_lv10_lgNone/pde_params_tsense/V0_config.json" &
run_evaluation "logs/klein_DM_10_lD1_lv1_lgNone/pde_params_tsense/V0_config.json" &
run_evaluation "logs/klein_PC_30_lD1_lv1_lgNone/pde_params_tsense/V0_config.json" &
run_evaluation logs/klein_DM_10_lD1_cfm1_lgNone_b512/pde_params_tsense/V0_config.json &
run_evaluation logs/klein_PC30_lD1_cfm1_lgNone/pde_params_tsense/V0_config.json


