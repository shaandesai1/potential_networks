#!/bin/bash

#mass spring
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 10 -tmax 3.05 -dt 0.05 -srate 0.05 -num_nodes 1 -dname pendulum -noise_std 0 -integrator rk4 -fname expt_a
python train.py -ni 10000 -n_test_traj 25 -n_train_traj 10 -tmax 3.05 -dt 0.05 -srate 0.05 -num_nodes 1 -dname pendulum -noise_std 0 -integrator vi4 -fname expt_a
