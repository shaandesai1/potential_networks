#!/bin/bash

#mass spring
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 5 -tmax 3.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -fname 5
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 10 -tmax 3.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -fname 10
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 100 -tmax 3.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -fname 100
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 1000 -tmax 3.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -fname 1000


