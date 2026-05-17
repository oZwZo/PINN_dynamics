# Aug 19
python main_train.py --config  logs/ery_mk_Aug1_multiscaled_n6/V0_Dpen_1.json  -G 0 & 
sleep 20s;

python main_train.py --config  logs/ery_mk_Aug1_multiscaled_n6/V1_Dpen_10.json  -G 0 & 
sleep 20s;

python main_train.py --config  logs/ery_mk_Aug1_multiscaled_n6/V2_weight_intens_0.json  -G 0 & 
sleep 20s;

python main_train.py --config  logs/ery_mk_Aug1_multiscaled_n6/V3_deltadm_weight_1.json  -G 0 & 
sleep 20s;

python main_train.py --config  logs/ery_mk_Aug1_multiscaled_n6/V4_deltadm_weight_10.json  -G 0 


