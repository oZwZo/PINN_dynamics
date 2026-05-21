#!/bin/bash

run_evaluation() {
    local config=$1
    # python=/local/scratch/wz369/PINN_env/bin/python
    python="singularity exec --nv /rds/user/wz369/hpc-work/containers/pseudodynamics+_torch2.10.0+cu128_20260304.sif python"
    pdp_dir=/rds/user/wz369/hpc-work/pseudodynamics_plus

    ode_log=${pdp_dir}/${config%.json}_eval_fate_ode.log
    sb_log=${pdp_dir}/${config%.json}_eval_fate_sb_sde.log
    sde_log=${pdp_dir}/${config%.json}_eval_fate_sde.log
    sb_log=${pdp_dir}/${config%.json}_eval_fate_sb_sde.log
    

    for time in 0.2 0.5 2.0 4.0 6.0; do

        echo "time: $time" >> $ode_log

        $python Explore_Notebook/14.Benmark_klein_addpop/2-1.eval_pdp_fate.py --sim_fn ode --config_path $pdp_dir/$config --t_end_norm $time --device cpu >> $ode_log

        for noise in 0.5 1 1.5 2.0; do
            echo "time: $time, noise: $noise" >> $sde_log
            $python Explore_Notebook/14.Benmark_klein_addpop/2-1.eval_pdp_fate.py --sim_fn sde --config_path $pdp_dir/$config --t_end_norm $time --noise_scale $noise >> $sde_log

            echo "time: $time, noise: $noise" >> $sb_log
            $python Explore_Notebook/14.Benmark_klein_addpop/2-1.eval_pdp_fate.py --sim_fn sb --config_path $pdp_dir/$config --t_end_norm $time --noise_scale $noise >> $sb_log
        done
    done
}
## DM
# # Non OT-assumption
# # run_evaluation "logs/klein_DM_10_lD10_lv1_lgNone/pde_params_tsense/V0_config.json" &
# run_evaluation "logs/klein_DM_10_lD1_lv10_lgNone/pde_params_tsense/V0_config.json" &
# run_evaluation "logs/klein_DM_10_lD1_lv1_lgNone/pde_params_tsense/V0_config.json" &
# wait

# # OT-assumption
# run_evaluation logs/klein_DM_10_lD1_cfm1_lgNone_b512/pde_params_tsense/V0_config.json &
# run_evaluation logs/klein_DM_10_lD1_cfm2_lgNone_b512/pde_params_tsense/V0_config.json &
# run_evaluation logs/klein_DM_10_lD1_cfm1_lv1_lgNone_b512/pde_params_tsense/V0_config.json &
# wait

# DMscaled OT-assumption
run_evaluation "logs/klein_DMscaled_10_cfm5_b512/pde_params_tsense/V0_config.json" &
run_evaluation "logs/klein_DMscaled_10_cfm5_b1024/pde_params_tsense/V0_config.json" &
wait

run_evaluation "logs/klein_DMscaled_10_cfm10_b512/pde_params_tsense/V0_config.json" &
run_evaluation "logs/klein_DMscaled_10_cfm10_b1024/pde_params_tsense/V0_config.json" &
wait

# ## PC
# # No OT-assumption
# run_evaluation "logs/klein_PC_30_lD1_lv1_lgNone/pde_params_tsense/V0_config.json" &

# # OT-assumption
# run_evaluation logs/klein_PC30_lD1_cfm1_lgNone/pde_params_tsense/V0_config.json & 
# run_evaluation logs/klein_PC30_lD1_cfm1_lv1_lgNone/pde_params_tsense/V0_config.json &
# run_evaluation logs/klein_PC30_lD1_cfm2_lgNone/pde_params_tsense/V0_config.json &
# wait

# run_evaluation logs/klein_PC30_lD1_cfm10_lgNone/pde_params_tsense/V0_config.json & 
# run_evaluation logs/klein_PC30_lD1_cfm10_lgNone_b1024/pde_params_tsense/V0_config.json & 
# run_evaluation logs/klein_DM_10_lD1_cfm10_lgNone_b512/pde_params_tsense/V0_config.json & 
# run_evaluation logs/klein_DM_10_lD1_cfm10_lgNone_b1024/pde_params_tsense/V0_config.json  
# wait
