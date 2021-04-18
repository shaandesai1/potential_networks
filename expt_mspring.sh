#!/bin/bash

#pendulum
for noisev in 0.1 0
do
for deltat in 0.1
do
  for INTEG_STEP in 10
  do
    for INTEG in rk4 vi4
    do
      for NONLIN in softplus
      do
        python train1.py -ni 5000 -long_range 1 -nonlinearity $NONLIN -hidden_dims 200 -num_hdims 2 -lr_iters 10000 -n_test_traj 25 -n_train_traj 25 -tmax 6 -dt $deltat -srate $deltat -num_nodes 1 -dname mass_spring -noise_std $noisev -integrator $INTEG -fname expt_a -integ_step $INTEG_STEP
      done
    done
  done
done
done
