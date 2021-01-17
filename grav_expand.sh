#!/bin/bash

#mass spring
python train.py -ni 20000 -n_test_traj 25 -n_train_traj 5 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -fname 5
python train.py -ni 20000 -n_test_traj 25 -n_train_traj 50 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -fname 50
python train.py -ni 20000 -n_test_traj 25 -n_train_traj 500 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -fname 500
python train.py -ni 20000 -n_test_traj 25 -n_train_traj 5000 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -fname 5000
