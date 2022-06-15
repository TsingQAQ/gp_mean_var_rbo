# python uncertainty_calibration_exp_.py -d 1 -nx_tr 10 -ff_n 50 100 150 200 250 -n_param 0.05 0.1 0.2 -r 8 -n_type 'uniform' -r_msr 'variance' -mc_i_num 500 -b_xt 10
# python uncertainty_calibration_exp_.py -d 2 -nx_tr 20 -ff_n 100 400 900 1600 2500 -n_param 0.05 0.1 0.2 -r 8 -n_type 'uniform' -r_msr 'variance' -mc_i_num 500 -b_xt 10
# python uncertainty_calibration_exp_.py -d 3 -nx_tr 30 -ff_n 1000 1728 2744 4096 5832 -n_param 0.05 0.1 0.2 -r 8 -n_type 'uniform' -r_msr 'variance' -mc_i_num 500 -b_xt 10
# python uncertainty_calibration_exp_.py -d 4 -nx_tr 40 -ff_n 625 1296 2401 4096 6561 -n_param 0.05 0.1 0.2 -r 8 -n_type 'uniform' -r_msr 'variance' -mc_i_num 500 -b_xt 10

# python uncertainty_calibration_exp_.py -d 1 -nx_tr 10 -ff_n 50 100 150 200 250 -n_param 0.001 0.005 0.01 -r 8 -n_type 'normal' -r_msr 'variance' -mc_i_num 500 -b_xt 10
# python uncertainty_calibration_exp_.py -d 2 -nx_tr 20 -ff_n 100 400 900 1600 2500 -n_param 0.001 0.005 0.01 -r 8 -n_type 'normal' -r_msr 'variance' -mc_i_num 500 -b_xt 10
python uncertainty_calibration_exp_.py -d 3 -nx_tr 30 -ff_n 1000 1728 2744 4096 5832 -n_param 0.001 0.005 0.01 -r 8 -n_type 'normal' -r_msr 'variance' -mc_i_num 500 -b_xt 10
# python uncertainty_calibration_exp_.py -d 4 -nx_tr 40 -ff_n 625 1296 2401 4096 6561 -n_param 0.001 0.005 0.01 -r 8 -n_type 'normal' -r_msr 'variance' -mc_i_num 500 -b_xt 10