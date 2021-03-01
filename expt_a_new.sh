#!/bin/bash

#pendulum
for noise_var in 0 0.1
do
  for deltat in 0.001 0.005 0.01 0.05 0.1
  do
    for INTEG_STEP in 2 5 10 20
    do
      for INTEG in rk1 rk2 rk3 rk4 vi1 vi2 vi3 vi4
      do
        for NONLIN in tanh softplus
        do
          python train.py -ni 2000 -long_range 1 -nonlinearity $NONLIN -hidden_dims 250 -num_hdims 2 -lr_iters 1000 -n_test_traj 25 -n_train_traj 20 -tmax 2 -dt $deltat -srate $deltat -num_nodes 1 -dname pendulum -noise_std $noise_var -integrator $INTEG -fname expt_a -integ_step $INTEG_STEP
        done
      done
    done
  done
done