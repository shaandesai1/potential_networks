from data_builder import *
from models import *
from utils import *
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('-hidden_dims', '--hidden_dims', type=int, default=200)
parser.add_argument('-num_hdims', '--num_hdims', type=int, default=2)
parser.add_argument('-tmax', '--tmax', type=float, default=3)
parser.add_argument('-lr_iters', '--lr_iters', type=int, default=10000)
parser.add_argument('-nonlinearity', '--nonlinearity', type=str, default='softplus')
parser.add_argument('-long_range', '--long_range', type=int, default=1)
parser.add_argument('-integ_step', '--integ_step', type=int, default=10)
parser.add_argument('-ni', '--num_iters', type=int, default=10000)
parser.add_argument("-n_test_traj", '--ntesttraj', type=int, default=25)
parser.add_argument('-srate', '--srate', type=float, default=0.1)
parser.add_argument('-dt', '--dt', type=float, default=0.1)
parser.add_argument('-integrator', '--integrator', type=str, default='rk4')
parser.add_argument('-save_name', '--save_name', type=str, default='trials_noise')
parser.add_argument('-dname', '--dname', type=str, default='mass_spring')
parser.add_argument('-num_nodes', '--num_nodes', type=int, default=1)
parser.add_argument('-noise_std', '--noise', type=float, default=0.1)
parser.add_argument('-fname', '--fname', type=str, default='expt_a')

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
n_train_traj = args.ntesttraj
T_max = args.tmax
T_max_t = 5 * T_max
dt = args.dt
srate = args.srate
# -1 due to down sampling

num_samples_per_traj = int(np.ceil((T_max / dt) / (srate / dt))) - 1
test_num_samples_per_traj = int(np.ceil((T_max_t / dt) / (srate / dt))) - 1

integrator_list = ['rk4', 'vi4']
if args.noise != 0:
    noisy = True
else:
    noisy = False
dataset_name = args.dname
expt_name = args.save_name
fname = f'{args.fname}_{args.hidden_dims}_{args.num_hdims}_{args.lr_iters}_{args.nonlinearity}_{args.dt}_{args.integ_step}'
test_data = get_dataset(dataset_name, expt_name, n_test_traj, num_nodes, T_max_t, dt, srate, 0, 3)

if args.long_range:
    BS = num_samples_per_traj
else:
    BS = 200

# dimension of a single particle, if 1D, spdim is 2
spdim = int(test_data['x'][0].shape[0] / num_nodes)
if args.long_range:
    print_every = 100
else:
    print_every = 1000

hamiltonian_fn = get_hamiltonian(dataset_name)
model_types = ['classic', 'graphic']

sublr = 1e-3
# model loop settings
model_types = ['classic', 'graphic']

classic_methods = ['dn', 'hnn', 'pnn']
graph_methods = ['dgn', 'hogn', 'pgn']

ener_coll = np.zeros((len(integrator_list), len(graph_methods) + len(classic_methods), n_test_traj))
err_coll = np.zeros((len(integrator_list), len(graph_methods) + len(classic_methods), n_test_traj))

state_collector = {}

for model_type in model_types:
    for integ_index, integ in enumerate(integrator_list):
        if model_type == 'classic':

            test_xnow = arrange_data(test_data, n_test_traj, num_nodes, T_max_t, dt, srate,
                                     spatial_dim=spdim, nograph=True, samp_size=int(np.ceil(T_max_t / dt)))
            for gm_index, classic_method in enumerate(classic_methods):

                data_dir = f'data/{dataset_name}/{classic_method}/{integ}/{args.integ_step}/{args.dt}/{fname}'
                tf.reset_default_graph()
                sess = tf.Session()
                kwargs = {'num_hdims': args.num_hdims,
                          'hidden_dims': args.hidden_dims,
                          'lr_iters': args.lr_iters,
                          'nonlinearity': args.nonlinearity,
                          'long_range': args.long_range,
                          'integ_step': args.integ_step}

                gm = nongraph_model(sess, classic_method, num_nodes, args.integ_step - 1, integ,
                                    expt_name, sublr, noisy, spdim, srate, **kwargs)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                if noisy:
                    saver.restore(sess, data_dir + classic_method + integ + fname + 'noisy')
                else:
                    saver.restore(sess, data_dir + classic_method + integ + fname)

                for t_iters in range(n_test_traj):
                    input_batch = test_xnow[0, t_iters, :].reshape(1, -1)
                    true_batch = test_xnow[1:, t_iters, :]
                    error, yhat = gm.test_step(input_batch, true_batch, test_xnow.shape[0] - 1)
                    yhat = yhat.reshape(-1, input_batch.shape[1])
                    true_batch = true_batch.reshape(-1, input_batch.shape[1])
                    # print(yhat.shape,true_batch.shape)
                    hp = hamiltonian_fn(yhat.squeeze(), model_type)
                    hp_gt = hamiltonian_fn(true_batch.squeeze(), model_type)
                    state_error = mean_squared_error(yhat, true_batch)
                    energy_error = mean_squared_error(np.sum(hp, 0), np.sum(hp_gt, 0))
                    test_std = ((yhat - true_batch) ** 2).std()
                    test_energy_std = ((np.sum(hp, 0) - np.sum(hp_gt, 0)) ** 2).std()

                    err_coll[integ_index, gm_index, t_iters] = state_error
                    ener_coll[integ_index, gm_index, t_iters] = energy_error
                    state_collector[f'{model_type}_{integ}_{t_iters}'] = yhat
                    state_collector[f'gt_{t_iters}'] = true_batch
        elif model_type == 'graphic':
            test_xnow = arrange_data(test_data, n_test_traj, num_nodes, T_max_t, dt, srate,
                                     spatial_dim=spdim, nograph=False, samp_size=int(np.ceil(T_max_t / dt)))

            test_mass = test_data['mass']
            test_ks = test_data['ks']

            kwargs = {'num_hdims': args.num_hdims,
                      'hidden_dims': args.hidden_dims,
                      'lr_iters': args.lr_iters,
                      'nonlinearity': args.nonlinearity,
                      'long_range': args.long_range,
                      'integ_step': args.integ_step}

            for gm_index, graph_method in enumerate(graph_methods):
                data_dir = f'data/{dataset_name}/{graph_method}/{integ}/{args.integ_step}/{args.dt}/{fname}'

                tf.reset_default_graph()
                sess = tf.Session()
                kwargs = {'num_hdims': args.num_hdims,
                          'hidden_dims': int(args.hidden_dims / 10),
                          'lr_iters': args.lr_iters,
                          'nonlinearity': args.nonlinearity,
                          'long_range': args.long_range,
                          'integ_step': args.integ_step}

                gm = graph_model(sess, graph_method, num_nodes, SAMPLE_BS,
                                 integ, expt_name, sublr, noisy, spdim, srate, **kwargs)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()

                saver.save(sess, data_dir + graph_method + integ + str(noisy))

                if noisy:
                    saver.restore(sess, data_dir + graphic_method + integ + fname + 'noisy')
                else:
                    saver.restore(sess, data_dir + graphic_method + integ + fname)

                for t_iters in range(n_test_traj):
                    input_batch = test_xnow[0, t_iters, :, :].reshape(-1, spdim)
                    true_batch = test_xnow[1:, t_iters, :, :].reshape(test_xnow.shape[0] - 1, -1, spdim)

                    error, yhat = gm.test_step(input_batch, true_batch, test_ks[t_iters].reshape(-1, num_nodes),
                                               test_mass[t_iters].reshape(-1, num_nodes), test_xnow.shape[0] - 1)
                    yhat = yhat.reshape(-1, spdim)
                    true_batch = true_batch.reshape(-1, spdim)
                    # print(yhat.shape)
                    hp = hamiltonian_fn(yhat.squeeze(), model_type)
                    hp_gt = hamiltonian_fn(true_batch.squeeze(), model_type)
                    state_error = mean_squared_error(yhat, true_batch)
                    energy_error = mean_squared_error(np.sum(hp, 0), np.sum(hp_gt, 0))
                    test_std = ((yhat - true_batch) ** 2).std()
                    test_energy_std = ((np.sum(hp, 0) - np.sum(hp_gt, 0)) ** 2).std()
                    err_coll[integ_index, gm_index + 3, t_iters] = state_error
                    ener_coll[integ_index, gm_index + 3, t_iters] = energy_error

                    state_collector[f'{model_type}_{integ}_{t_iters}'] = yhat
                    state_collector[f'gt_{t_iters}'] = true_batch

if noisy:
    np.save(f'{dataset_name}_state_error_noisy.npy',err_coll)
    np.save(f'{dataset_name}_energy_error_noisy.npy',ener_coll)

    with open(f'{dataset_name}_state_collector_noisy.pickle', 'wb') as handle:
        pickle.dump(state_collector, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    np.save(f'{dataset_name}_state_error.npy',err_coll)
    np.save(f'{dataset_name}_energy_error.npy',ener_coll)

    with open(f'{dataset_name}_state_collector.pickle', 'wb') as handle:
        pickle.dump(state_collector, handle, protocol=pickle.HIGHEST_PROTOCOL)


