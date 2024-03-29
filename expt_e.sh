#!/bin/bash

#mass spring
python train.py -ni 50000 -n_test_traj 25 -n_train_traj 25 -tmax 4.1 -dt 0.1 -srate 0.1 -num_nodes 5 -dname n_spring -noise_std 0 -integrator rk4 -fname expt_e
python train.py -ni 50000 -n_test_traj 25 -n_train_traj 25 -tmax 4.1 -dt 0.1 -srate 0.1 -num_nodes 5 -dname n_spring -noise_std 0 -integrator vi4 -fname expt_e
