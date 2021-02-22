#!/bin/bash

#mass spring
python train.py -ni 500 -n_test_traj 25 -n_train_traj 10 -tmax 5.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname mass_spring -noise_std 0.1 -integrator rk2 -fname expt_a_testsht
python train.py -ni 500 -n_test_traj 25 -n_train_traj 10 -tmax 5.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname mass_spring -noise_std 0.1 -integrator vi2 -fname expt_a_testsht
python train.py -ni 500 -n_test_traj 25 -n_train_traj 10 -tmax 5.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname mass_spring -noise_std 0.1 -integrator rk4 -fname expt_a_testsht
python train.py -ni 500 -n_test_traj 25 -n_train_traj 10 -tmax 5.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname mass_spring -noise_std 0.1 -integrator vi4 -fname expt_a_testsht

#python train_pyc_long.py -ni 20000 -n_test_traj 20 -n_train_traj 20 -tmax 4.1 -dt 0.1 -srate 0.1 -num_nodes 5 -dname n_spring -noise_std 0 -integrator rk4 -fname expt_a
#python train_pyc_long.py -ni 20000 -n_test_traj 20 -n_train_traj 20 -tmax 4.1 -dt 0.1 -srate 0.1 -num_nodes 5 -dname n_spring -noise_std 0 -integrator vi4 -fname expt_a
#python train_pyc.py -ni 10000 -n_test_traj 25 -n_train_traj 10 -tmax 3.05 -dt 0.05 -srate 0.05 -num_nodes 1 -dname pendulum -noise_std 0 -integrator rk4 -fname expt_a
#python train_pyc.py -ni 10000 -n_test_traj 25 -n_train_traj 10 -tmax 3.05 -dt 0.05 -srate 0.05 -num_nodes 1 -dname pendulum -noise_std 0 -integrator vi4 -fname expt_a
#python train_pyc.py -ni 10000 -n_test_traj 25 -n_train_traj 10 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator rk4 -fname expt_a
#python train_pyc.py -ni 10000 -n_test_traj 25 -n_train_traj 10 -tmax 20.4 -dt 0.4 -srate 0.4 -num_nodes 2 -dname n_grav -noise_std 0 -integrator vi4 -fname expt_a
