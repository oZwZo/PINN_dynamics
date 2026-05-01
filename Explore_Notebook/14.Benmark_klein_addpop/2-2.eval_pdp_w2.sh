#!/bin/bash

run_evaluation() {
    local config=$1
    # python=/local/scratch/wz369/PINN_env/bin/python
    python=/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python
    pdp_dir=/rds/user/wz369/hpc-work/pseudodynamics_plus

    ode_log=${pdp_dir}/${config%.json}_w2_ode.log
    sde_log=${pdp_dir}/${config%.json}_w2_sde.log
    sb_log=${pdp_dir}/${config%.json}_w2_sb_sde.log
    

    for time in 2.5 3.0 3.5 4.0; do

        echo "time: $time" >> $ode_log

        $python Explore_Notebook/14.Benmark_klein_addpop/2-2.eval_pdp_w2.py --device cpu --sim_fn ode --config_path $pdp_dir/$config --t_end_norm $time >> $ode_log

        for noise in 0.5 1 1.5 2.0; do
            echo "time: $time, noise: $noise" >> $sde_log
            $python Explore_Notebook/14.Benmark_klein_addpop/2-2.eval_pdp_w2.py --device cpu --sim_fn sde --config_path $pdp_dir/$config --t_end_norm $time --noise_scale $noise >> $sde_log

            echo "time: $time, noise: $noise" >> $sb_log
            $python Explore_Notebook/14.Benmark_klein_addpop/2-2.eval_pdp_w2.py --device cpu --sim_fn sb --config_path $pdp_dir/$config --t_end_norm $time --noise_scale $noise >> $sb_log
        done
    done
}


# ## PC
# # No OT-assumption
run_evaluation "logs/klein_PC_30_lD1_lv1_lgNone/pde_params_tsense/V0_config.json" 

# # OT-assumption
run_evaluation logs/klein_PC30_lD1_cfm1_lgNone/pde_params_tsense/V0_config.json & 
run_evaluation logs/klein_PC30_lD1_cfm1_lv1_lgNone/pde_params_tsense/V0_config.json &
wait

run_evaluation logs/klein_PC30_lD1_cfm2_lgNone/pde_params_tsense/V0_config.json &
run_evaluation logs/klein_PC30_lD1_cfm10_lgNone/pde_params_tsense/V0_config.json & 
run_evaluation logs/klein_PC30_lD1_cfm10_lgNone_b1024/pde_params_tsense/V0_config.json & 

wait

# ## DM
# # Non OT-assumption
run_evaluation "logs/klein_DM_10_lD10_lv1_lgNone/pde_params_tsense/V0_config.json" &
run_evaluation "logs/klein_DM_10_lD1_lv10_lgNone/pde_params_tsense/V0_config.json" &
run_evaluation "logs/klein_DM_10_lD1_lv1_lgNone/pde_params_tsense/V0_config.json" &


# # # OT-assumption
run_evaluation logs/klein_DM_10_lD1_cfm1_lgNone_b512/pde_params_tsense/V0_config.json &
wait

run_evaluation logs/klein_DM_10_lD1_cfm2_lgNone_b512/pde_params_tsense/V0_config.json &
run_evaluation logs/klein_DM_10_lD1_cfm1_lv1_lgNone_b512/pde_params_tsense/V0_config.json &
run_evaluation logs/klein_DM_10_lD1_cfm10_lgNone_b512/pde_params_tsense/V0_config.json & 
run_evaluation logs/klein_DM_10_lD1_cfm10_lgNone_b1024/pde_params_tsense/V0_config.json  &
wait
