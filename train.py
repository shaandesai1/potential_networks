"""
Author: ***
Code to produce the results obtained in VIGN: Variational Integrator Graph Networks

"""

from data_builder import *
from models import *
from utils import *
from tensorboardX import SummaryWriter
import argparse
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('-hidden_dims', '--hidden_dims', type=float, default=256)
parser.add_argument('-num_hdims', '--num_hdims', type=int, default=2)
parser.add_argument('-lr_iters', '--lr_iters', type=float, default=10000)
parser.add_argument('-nonlinearity', '--nonlinearity', type=str, default='tanh')
parser.add_argument('-long_range', '--long_range', type=int, default=0)
parser.add_argument('-integ_step', '--integ_step', type=int, default=2)

parser.add_argument('-ni', '--num_iters', type=int, default=10000)
parser.add_argument("-n_test_traj", '--ntesttraj', type=int, default=20)
parser.add_argument("-n_train_traj", '--ntraintraj', type=int, default=10)
parser.add_argument('-srate', '--srate', type=float, default=0.4)
parser.add_argument('-dt', '--dt', type=float, default=0.4)
parser.add_argument('-tmax', '--tmax', type=float, default=20.4)
parser.add_argument('-integrator', '--integrator', type=str, default='rk4')
parser.add_argument('-save_name', '--save_name', type=str, default='trials_noise')
parser.add_argument('-num_nodes', '--num_nodes', type=int, default=2)
parser.add_argument('-dname', '--dname', type=str, default='n_grav')
parser.add_argument('-noise_std', '--noise', type=float, default=0)
parser.add_argument('-fname', '--fname', type=str, default='a')

verbose = True
verbose1 = False
args = parser.parse_args()

if args.long_range == 0:
    args.long_range = False
else:
    args.long_range = True

# print(args.long_range)

num_nodes = args.num_nodes
iters = args.num_iters
n_test_traj = args.ntesttraj
num_trajectories = args.ntraintraj
T_max = args.tmax
T_max_t = T_max * 3
dt = args.dt
srate = args.srate
# -1 due to down sampling

num_samples_per_traj = int(np.ceil((T_max / dt) / (srate / dt))) - 1
test_num_samples_per_traj = int(np.ceil((T_max_t / dt) / (srate / dt))) - 1

integ = args.integrator
if args.noise != 0:
    noisy = True
else:
    noisy = False
dataset_name = args.dname
expt_name = args.save_name
fname = f'{args.fname}_{args.hidden_dims}_{args.num_hdims}_{args.lr_iters}_{args.nonlinearity}_{args.long_range}_{args.dt}'
# dataset preprocessing
train_data = get_dataset(dataset_name, expt_name, num_trajectories, num_nodes, T_max, dt, srate, args.noise, 0)
valid_data = get_dataset(dataset_name, expt_name, n_test_traj, num_nodes, T_max_t, dt, srate, 0, 11)

if args.long_range:
    BS = num_samples_per_traj
else:
    BS = 200
BS_test = test_num_samples_per_traj
# dimension of a single particle, if 1D, spdim is 2
spdim = int(train_data['x'][0].shape[0] / num_nodes)
if args.long_range:
    print_every = 10
else:
    print_every = 1000

hamiltonian_fn = get_hamiltonian(dataset_name)
# model loop settings
model_types = ['graphic']

classic_methods = ['dn', 'hnn', 'pnn']
graph_methods = ['dgn', 'hogn', 'pgn']

sublr = 1e-3
df_all = pd.DataFrame(
    columns=['model', 'model_type', 'sample',
             'train_state_error', 'train_state_std', 'train_energy_error', 'train_energy_std',
             'valid_state_error', 'valid_std', 'valid_energy_error', 'valid_energy_std',
             'test_state_error', 'test_std', 'test_energy_error', 'test_energy_std'])

for model_type in model_types:
    if model_type == 'classic':
        xnow = arrange_data(train_data, num_trajectories, num_nodes, T_max, dt, srate,
                            spatial_dim=spdim, nograph=True, samp_size=args.integ_step)
        test_xnow = arrange_data(valid_data, n_test_traj, num_nodes, T_max_t, dt, srate,
                                 spatial_dim=spdim, nograph=True, samp_size=int(np.ceil(T_max_t / dt)))

        num_training_iterations = iters

        for gm_index, classic_method in enumerate(classic_methods):

            data_dir = 'data/' + dataset_name + '/' + str(sublr) + '/' + classic_method + '/' + fname + '/'
            if not os.path.exists(data_dir):
                print('non existent path....creating path')
                os.makedirs(data_dir)
            dirw = classic_method + str(sublr) + integ + fname
            if noisy:
                writer = SummaryWriter(
                    'noisy/' + dataset_name + '/' + str(sublr) + '/' + classic_method + '/' + dirw)
            else:
                writer = SummaryWriter(
                    'noiseless/' + dataset_name + '/' + str(sublr) + '/' + classic_method + '/' + dirw)
            try:
                sess.close()
            except NameError:
                pass

            tf.reset_default_graph()
            sess = tf.Session()
            kwargs = {'num_hdims': args.num_hdims,
                      'hidden_dims': args.hidden_dims,
                      'lr_iters': args.lr_iters,
                      'nonlinearity': args.nonlinearity,
                      'long_range': args.long_range,
                      'integ_step': args.integ_step}

            gm = nongraph_model(sess, classic_method, num_nodes, xnow.shape[0] - 1, integ,
                                expt_name, sublr, noisy, spdim, srate, **kwargs)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            for iteration in range(num_training_iterations):
                for sub_iter in range(1):
                    input_batch = xnow[0, :, :]
                    true_batch = xnow[1:, :, :]
                    loss, _ = gm.train_step(input_batch, true_batch)
                    # print(iteration * Tot_iters + sub_iter)
                    writer.add_scalar('train_loss', loss, iteration + sub_iter)
                    if ((iteration + sub_iter) % print_every == 0):
                        print('Iteration:{},Training Loss:{:.3g}'.format(iteration + sub_iter, loss))

            for t_iters in range(n_test_traj):
                input_batch = test_xnow[0, t_iters, :].reshape(1, -1)
                true_batch = test_xnow[1:, t_iters, :]
                error, yhat = gm.test_step(input_batch, true_batch, BS_test)
                hp = hamiltonian_fn(yhat, model_type)
                hp_gt = hamiltonian_fn(true_batch, model_type)
                state_error = mean_squared_error(yhat, true_batch)
                energy_error = mean_squared_error(np.sum(hp, 0), np.sum(hp_gt, 0))
                test_std = ((yhat - true_batch) ** 2).std()
                test_energy_std = ((np.sum(hp, 0) - np.sum(hp_gt, 0)) ** 2).std()
                df_all.loc[len(df_all)] = [classic_method, model_type, t_iters,
                                           1, 1, 1, 1,
                                           1, 1, 1, 1,
                                           state_error, test_std, energy_error, test_energy_std]
                df_all.to_csv(f'run_data_{dataset_name}_{integ}_{noisy}_{fname}.csv')

    elif model_type == 'graphic':
        xnow = arrange_data(train_data, num_trajectories, num_nodes, T_max, dt, srate,
                            spatial_dim=spdim, nograph=False, samp_size=args.integ_step)
        test_xnow = arrange_data(valid_data, n_test_traj, num_nodes, T_max_t, dt, srate,
                                 spatial_dim=spdim, nograph=False, samp_size=int(np.ceil(T_max_t / dt)))

        newmass = np.repeat(train_data['mass'], int(np.ceil(T_max / dt))-args.integ_step+1, axis=0).reshape(-1,num_nodes)
        newks = np.repeat(train_data['ks'], int(np.ceil(T_max / dt))-args.integ_step+1, axis=0).reshape(-1,num_nodes)
        test_mass = valid_data['mass']
        test_ks = valid_data['ks']

        num_training_iterations = iters

        kwargs = {'num_hdims': args.num_hdims,
                  'hidden_dims': args.hidden_dims,
                  'lr_iters': args.lr_iters,
                  'nonlinearity': args.nonlinearity,
                  'long_range': args.long_range,
                  'integ_step': args.integ_step}


        for gm_index, graph_method in enumerate(graph_methods):
            data_dir = 'data/' + dataset_name + '/' + str(sublr) + '/' + graph_method + '/' + fname + '/'
            if not os.path.exists(data_dir):
                print('non existent path....creating path')
                os.makedirs(data_dir)

            dirw = graph_method + str(sublr) + integ + fname
            if noisy:
                writer = SummaryWriter('noisy/' + dataset_name + '/' + str(sublr) + '/' + graph_method + '/' + dirw)
            else:
                writer = SummaryWriter(
                    'noiseless/' + dataset_name + '/' + str(sublr) + '/' + graph_method + '/' + dirw)

            try:
                sess.close()
            except NameError:
                pass

            tf.reset_default_graph()
            sess = tf.Session()
            kwargs = {'num_hdims': args.num_hdims,
                      'hidden_dims': args.hidden_dims,
                      'lr_iters': args.lr_iters,
                      'nonlinearity': args.nonlinearity,
                      'long_range': args.long_range,
                      'integ_step': args.integ_step}

            gm = graph_model(sess, graph_method, num_nodes, xnow.shape[1],
                 integ, expt_name, sublr, noisy, spdim, srate, **kwargs)
            sess.run(tf.global_variables_initializer())


            saver = tf.train.Saver()
            for iteration in range(num_training_iterations):
                for sub_iter in range(1):
                    # samp_size, -1, num_nodes, 2vdim
                    input_batch = xnow[0, :, :, :].reshape(-1,spdim)
                    true_batch = xnow[1:, :, :, :].reshape(args.integ_step-1,-1,spdim)
                    print(input_batch.shape)
                    print(true_batch.shape)
                    print(newmass.shape)
                    print(xnow.shape[1])
                    loss, _ = gm.train_step(input_batch, true_batch,newmass,newks)

                    # writer.add_scalar('train_loss', loss, iteration * Tot_iters + sub_iter)
                    if ((iteration) % print_every == 0) and verbose == True:
                        print('Iteration:{},Training Loss:{:.3g}'.format(iteration + sub_iter, loss))

            print('Iteration:{},Training Loss:{:.3g}'.format(iteration  + sub_iter, loss))

            for t_iters in range(n_test_traj):
                input_batch = test_xnow[num_nodes * t_iters * BS_test:num_nodes * t_iters * BS_test + num_nodes]
                true_batch = test_xnext[num_nodes * t_iters * BS_test:num_nodes * (t_iters + 1) * BS_test]
                error, yhat = gm.test_step(input_batch, true_batch, np.reshape(test_ks[t_iters * BS_test], [1, -1]),
                                           np.reshape(test_mass[t_iters * BS_test], [1, -1]), BS_test)
                hp = hamiltonian_fn(yhat, model_type)
                hp_gt = hamiltonian_fn(true_batch, model_type)
                state_error = mean_squared_error(yhat, true_batch)
                energy_error = mean_squared_error(np.sum(hp, 0), np.sum(hp_gt, 0))
                test_std = ((yhat - true_batch) ** 2).std()
                test_energy_std = ((np.sum(hp, 0) - np.sum(hp_gt, 0)) ** 2).std()
                df_all.loc[len(df_all)] = [graph_method, model_type, t_iters,
                                           1, 1, 1, 1,
                                           1, 1, 1, 1,
                                           state_error, test_std, energy_error, test_energy_std]

                df_all.to_csv(f'run_data_{dataset_name}_{integ}_{noisy}_{fname}.csv')

    try:
        sess.close()
    except NameError:
        print('NameError')

    df_all.to_csv(f'run_data_{dataset_name}_{integ}_{noisy}_{fname}.csv')
