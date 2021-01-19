#!/bin/bash

#mass spring
python train.py -ni 20000 -n_test_traj 25 -n_train_traj 10 -tmax 3.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -integrator rk4
python train.py -ni 20000 -n_test_traj 25 -n_train_traj 10 -tmax 3.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -integrator vi4
