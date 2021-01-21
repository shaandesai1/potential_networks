#!/bin/bash

#mass spring
python train.py -ni 20000 -n_test_traj 25 -n_train_traj 5 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -fname 5_long
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 5 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -fname 5 -integrator rk4

#python train.py -ni 10000 -n_test_traj 25 -n_train_traj 10 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -fname 10
#python train.py -ni 10000 -n_test_traj 25 -n_train_traj 100 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -fname 100
#python train.py -ni 10000 -n_test_traj 25 -n_train_traj 1000 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -fname 1000
#python train.py -ni 20000 -n_test_traj 25 -n_train_traj 5000 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -fname 5000
