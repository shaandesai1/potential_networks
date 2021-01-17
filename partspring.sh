#!/bin/bash

#mass spring
python train.py -ni 100000 -n_test_traj 20 -n_train_traj 100 -tmax 4.1 -dt 0.1 -srate 0.1 -num_nodes 5 -dname n_spring -noise_std 0
