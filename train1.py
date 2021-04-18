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

parser.add_argument('-hidden_dims', '--hidden_dims', type=int, default=256)
parser.add_argument('-num_hdims', '--num_hdims', type=int, default=2)
parser.add_argument('-lr_iters', '--lr_iters', type=int, default=10000)
parser.add_argument('-nonlinearity', '--nonlinearity', type=str, default='tanh')
parser.add_argument('-long_range', '--long_range', type=int, default=0)
parser.add_argument('-integ_step', '--integ_step', type=int, default=2)
# parser.add_argument('-')
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
T_max_t = 2*T_max
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
fname = f'{args.fname}_{args.hidden_dims}_{args.num_hdims}_{args.lr_iters}_{args.nonlinearity}_{args.dt}_{args.integ_step}'
# dataset preprocessing
train_data = get_dataset(dataset_name, expt_name, num_trajectories, num_nodes, T_max, dt, srate, args.noise, 0)
valid_data = get_dataset(dataset_name, expt_name, num_trajectories, num_nodes, T_max, dt, srate, 0, 1)
test_data = get_dataset(dataset_name, expt_name, n_test_traj, num_nodes, T_max_t, dt, srate, 0, 2)

if args.long_range:
    BS = num_samples_per_traj
else:
    BS = 200
BS_test = test_num_samples_per_traj
# dimension of a single particle, if 1D, spdim is 2
spdim = int(train_data['x'][0].shape[0] / num_nodes)
if args.long_range:
    print_every = 100
else:
    print_every = 1000

hamiltonian_fn = get_hamiltonian(dataset_name)
# model loop settings
model_types = ['classic','graphic']

classic_methods = ['dn', 'hnn', 'pnn']
graph_methods = ['dgn', 'hogn', 'pgn']

sublr = 1e-3
df_all = pd.DataFrame(
    columns=['model', 'model_type', 'sample',
             'train_state_error', 'train_state_std', 'train_energy_error', 'train_energy_std',
             'valid_state_error', 'valid_std', 'valid_energy_error', 'valid_energy_std',
             'test_state_error', 'test_std', 'test_energy_error', 'test_energy_std'])

SAMPLE_BS = 200


for model_type in model_types:
    if model_type == 'classic':
        xnow = arrange_data(train_data, num_trajectories, num_nodes, T_max, dt, srate,
                            spatial_dim=spdim, nograph=True, samp_size=args.integ_step)

        valid_xnow = arrange_data(valid_data, num_trajectories, num_nodes, T_max, dt, srate,
                                 spatial_dim=spdim, nograph=True, samp_size=args.integ_step)

        test_xnow = arrange_data(test_data, n_test_traj, num_nodes, T_max_t, dt, srate,
                                 spatial_dim=spdim, nograph=True, samp_size=int(np.ceil(T_max_t / dt)))

        num_training_iterations = iters

        for gm_index, classic_method in enumerate(classic_methods):

            data_dir = f'data/{dataset_name}/{classic_method}/{integ}/{fname}'
            if not os.path.exists(data_dir):
                print('non existent path....creating path')
                os.makedirs(data_dir)
            if noisy:
                writer = SummaryWriter(f'noisy/{dataset_name}/{classic_method}/{integ}/{fname}')
            else:
                writer = SummaryWriter(f'noiseless/{dataset_name}/{classic_method}/{integ}/{fname}')
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
            bsv = np.arange(xnow.shape[1])
            # np.random.shuffle()
            for iteration in range(num_training_iterations):
                for sub_iter in range(1):
                    np.random.shuffle(bsv)
                    bss = bsv[:SAMPLE_BS]
                    # print(bss)
                    input_batch = xnow[0, bss, :]
                    true_batch = xnow[1:, bss, :]
                    loss, _ = gm.train_step(input_batch, true_batch)
                    # writer.add_scalar('train_loss', loss, iteration + sub_iter)
                    # if ((iteration + sub_iter) % print_every == 0):
                    #     print('Iteration:{},Training Loss:{:.3g}'.format(iteration + sub_iter, loss))
                    #     input_batch = valid_xnow[0,bss,:]
                    #     true_batch = valid_xnow[1:,bss,:]
                    #     # t1_start = process_time()
                    #     loss, _ = gm.valid_step(input_batch, true_batch)
                    #     print('Iteration:{},Validation Loss:{:.3g}'.format(iteration + sub_iter, loss))
                    #     writer.add_scalar('valid_loss', loss, iteration + sub_iter)
            saver.save(sess, data_dir + graph_method + integ + str(noisy))

    elif model_type == 'graphic':
        xnow = arrange_data(train_data, num_trajectories, num_nodes, T_max, dt, srate,
                            spatial_dim=spdim, nograph=False, samp_size=args.integ_step)
        valid_xnow = arrange_data(valid_data, num_trajectories, num_nodes, T_max, dt, srate,
                                 spatial_dim=spdim, nograph=False, samp_size=args.integ_step)
        test_xnow = arrange_data(test_data, n_test_traj, num_nodes, T_max_t, dt, srate,
                                 spatial_dim=spdim, nograph=False, samp_size=int(np.ceil(T_max_t / dt)))

        newmass = np.repeat(train_data['mass'], int(np.ceil(T_max / dt))-args.integ_step+1, axis=0).reshape(-1,num_nodes)
        newks = np.repeat(train_data['ks'], int(np.ceil(T_max / dt))-args.integ_step+1, axis=0).reshape(-1,num_nodes)

        valid_mass = np.repeat(valid_data['mass'], int(np.ceil(T_max / dt)) - args.integ_step + 1, axis=0).reshape(-1,
                                                                                                                num_nodes)
        valid_ks = np.repeat(valid_data['ks'], int(np.ceil(T_max / dt)) - args.integ_step + 1, axis=0).reshape(-1,
                                                                                                            num_nodes)

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
            data_dir = f'data/{dataset_name}/{graph_method}/{integ}/{fname}'
            if not os.path.exists(data_dir):
                print('non existent path....creating path')
                os.makedirs(data_dir)
            if noisy:
                writer = SummaryWriter(f'noisy/{dataset_name}/{graph_method}/{integ}/{fname}')
            else:
                writer = SummaryWriter(f'noiseless/{dataset_name}/{graph_method}/{integ}/{fname}')
            try:
                sess.close()
            except NameError:
                pass

            tf.reset_default_graph()
            sess = tf.Session()
            kwargs = {'num_hdims': args.num_hdims,
                      'hidden_dims': int(args.hidden_dims/10),
                      'lr_iters': args.lr_iters,
                      'nonlinearity': args.nonlinearity,
                      'long_range': args.long_range,
                      'integ_step': args.integ_step}


            # print(f'xnowshape:{xnow.shape}')
            gm = graph_model(sess, graph_method, num_nodes, SAMPLE_BS,
                 integ, expt_name, sublr, noisy, spdim, srate, **kwargs)
            sess.run(tf.global_variables_initializer())


            saver = tf.train.Saver()

            bsv = np.arange(xnow.shape[1])
            for iteration in range(num_training_iterations):
                for sub_iter in range(1):
                    np.random.shuffle(bsv)
                    bss = bsv[:SAMPLE_BS]
                    # samp_size, -1, num_nodes, 2vdim
                    input_batch = xnow[0, bss, :, :].reshape(-1,spdim)
                    true_batch = xnow[1:, bss, :, :].reshape(args.integ_step-1,-1,spdim)
                    loss, _ = gm.train_step(input_batch, true_batch,newmass[bss],newks[bss])
                    # writer.add_scalar('train_loss', loss, iteration + sub_iter)
                    # if ((iteration + sub_iter) % print_every == 0):
                    #     print('Iteration:{},Training Loss:{:.3g}'.format(iteration + sub_iter, loss))
                    #     input_batch = valid_xnow[0, bss,:, :].reshape(-1,spdim)
                    #     true_batch = valid_xnow[1:, bss, :,:].reshape(args.integ_step-1,-1,spdim)
                    #     # t1_start = process_time()
                    #     loss, _ = gm.valid_step(input_batch, true_batch,valid_mass[bss],valid_ks[bss])
                    #     print('Iteration:{},Validation Loss:{:.3g}'.format(iteration + sub_iter, loss))
                    #     writer.add_scalar('valid_loss', loss, iteration + sub_iter)

#print('Iteration:{},Training Loss:{:.3g}'.format(iteration  + sub_iter, loss))

            saver.save(sess, data_dir + graph_method + integ + str(noisy))
    try:
        sess.close()
    except NameError:
        print('NameError')

    # df_all.to_csv(f'run_data_{dataset_name}_{integ}_{noisy}_{fname}.csv')
