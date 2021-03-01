#!/bin/bash

#pendulum

for deltat in 0.05
do
  for INTEG_STEP in 3
  do
    for INTEG in rk4
    do
      for NONLIN in softplus
      do
        python train.py -ni 2000 -long_range 1 -nonlinearity $NONLIN -hidden_dims 200 -num_hdims 2 -lr_iters 1000 -n_test_traj 25 -n_train_traj 20 -tmax 2 -dt $deltat -srate $deltat -num_nodes 2 -dname n_grav -noise_std 0 -integrator $INTEG -fname expt_a -integ_step $INTEG_STEP
      done
    done
  done
done
