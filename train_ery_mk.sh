# python main_train.py --config logs/ery_mk_Aug1_debug/pde_params_tsense/V0_Dpen_1.json -G 0

# Aug 19
python main_train.py --config logs/ery_mk_Aug1_multiscaled_n7_minminus/V0_Dpen_1.json -G 0 &
sleep 10s;

python main_train.py --config logs/ery_mk_Aug1_multiscaled_n7_minminus/V1_Dpen_10.json -G 0 &
sleep 10s;

python main_train.py --config logs/ery_mk_Aug1_multiscaled_n7_minminus/V2_weight_intens_1.json -G 0 &
sleep 10s;

python main_train.py --config logs/ery_mk_Aug1_multiscaled_n7_minminus/V3_weight_intens_neg1.json -G 0 &
sleep 10s;

python main_train.py --config logs/ery_mk_Aug1_multiscaled_n7_minminus/V4_deltadm_weight_1.json -G 0

