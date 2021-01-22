#!/bin/bash

python train.py -ni 10000 -n_test_traj 25 -n_train_traj 5 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator rk4 -fname expt_c_5
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 5 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator vi4 -fname expt_c_5
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 20 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator rk4 -fname expt_c_20
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 20 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator vi4 -fname expt_c_20
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 100 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator rk4 -fname expt_c_100
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 100 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator vi4 -fname expt_c_100
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 200 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator rk4 -fname expt_c_200
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 200 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator vi4 -fname expt_c_200
