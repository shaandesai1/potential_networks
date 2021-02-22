#!/bin/bash

#mass spring

python train.py -ni 20000 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 10000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -integrator rk2 -fname expt_a
python train.py -ni 20000 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 10000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -integrator vi2 -fname expt_a
python train.py -ni 20000 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 10000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -integrator rk4 -fname expt_a
python train.py -ni 20000 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 10000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -integrator vi4 -fname expt_a

python train.py -ni 20000 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 10000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0.1 -integrator rk2 -fname expt_a
python train.py -ni 20000 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 10000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0.1 -integrator vi2 -fname expt_a
python train.py -ni 20000 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 10000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0.1 -integrator rk4 -fname expt_a
python train.py -ni 20000 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 10000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0.1 -integrator vi4 -fname expt_a

python train.py -ni 2000 -long_range 1 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 1000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -integrator rk2 -fname expt_a
python train.py -ni 2000 -long_range 1 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 1000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -integrator vi2 -fname expt_a
python train.py -ni 5000 -long_range 1 -nonlinearity tanh -hidden_dims 100 -num_hdims 3 -lr_iters 5000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -integrator rk4 -fname expt_a
python train.py -ni 2000 -long_range 1 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 1000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0 -integrator vi4 -fname expt_a

python train.py -ni 2000 -long_range 1 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 1000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0.1 -integrator rk2 -fname expt_a
python train.py -ni 2000 -long_range 1 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 1000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0.1 -integrator vi2 -fname expt_a
python train.py -ni 2000 -long_range 1 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 1000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0.1 -integrator rk4 -fname expt_a
python train.py -ni 2000 -long_range 1 -nonlinearity softplus -hidden_dims 250 -num_hdims 2 -lr_iters 1000 -n_test_traj 25 -n_train_traj 20 -tmax 2.1 -dt 0.1 -srate 0.1 -num_nodes 1 -dname pendulum -noise_std 0.1 -integrator vi4 -fname expt_a


