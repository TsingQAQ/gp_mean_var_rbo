python mc_robustness_accuracy_compare.py -d 1 -cfg 'SinLinear' -nx_tr 20 -ff_n 50 100 150 200 250 -n_param 0.05 0.1 0.2 -r 64 -n_type 'uniform' -r_msr 'variance' -mc_i_num 500 -b_xt 50
python mc_robustness_accuracy_compare.py -d 2 -cfg 'GMM' -nx_tr 40 -ff_n 100 400 900 1600 2500 -n_param 0.05 0.1 0.2 -r 64 -n_type 'uniform' -r_msr 'variance' -mc_i_num 500 -b_xt 30
python mc_robustness_accuracy_compare.py -d 3 -cfg 'Hartmann3' -nx_tr 60 -ff_n 1000 1728 2744 4096 5832 -n_param 0.05 0.1 0.2 -r 64 -n_type 'uniform' -r_msr 'variance' -mc_i_num 500 -b_xt 10

python mc_robustness_accuracy_compare.py -d 1 -cfg 'SinLinear' -nx_tr 20 -ff_n 50 100 150 200 250 -n_param 0.001 0.005 0.01 -r 64 -n_type 'normal' -r_msr 'variance' -mc_i_num 500 -b_xt 50
python mc_robustness_accuracy_compare.py -d 2 -cfg 'GMM' -nx_tr 40 -ff_n 100 400 900 1600 2500 -n_param 0.001 0.005 0.01 -r 64 -n_type 'normal' -r_msr 'variance' -mc_i_num 500 -b_xt 30
python mc_robustness_accuracy_compare.py -d 3 -cfg 'Hartmann3' -nx_tr 60 -ff_n 1000 1728 2744 4096 5832 -n_param 0.001 0.005 0.01 -r 64 -n_type 'normal' -r_msr 'variance' -mc_i_num 500 -b_xt 10