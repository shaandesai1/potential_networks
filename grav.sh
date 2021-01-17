#!/bin/bash

#mass spring
python train.py -ni 20000 -n_test_traj 20 -n_train_traj 25 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator rk4
python train.py -ni 20000 -n_test_traj 20 -n_train_traj 25 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator vi4
