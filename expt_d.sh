#!/bin/bash

#mass spring
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 25 -tmax 6.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname mass_spring -noise_std 0 -integrator rk4 -fname expt_d
#python train.py -ni 20000 -n_test_traj 25 -n_train_traj 25 -tmax 6.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname mass_spring -noise_std 0 -integrator vi4 -fname expt_d
