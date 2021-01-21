#!/bin/bash

#mass spring
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 10 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator rk4 -fname expt_a
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 10 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator vi4 -fname expt_a
