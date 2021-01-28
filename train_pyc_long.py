"""
Author: ***
Code to produce the results obtained in VIGN: Variational Integrator Graph Networks

"""

from data_builder import *
from models_pyc_long import *
from utils import *
from tensorboardX import SummaryWriter
import argparse
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-ni', '--num_iters', type=int, default=10000)
parser.add_argument("-n_test_traj", '--ntesttraj', type=int, default=20)
parser.add_argument("-n_train_traj", '--ntraintraj', type=int, default=10)
parser.add_argument('-srate', '--srate', type=float, default=0.4)
parser.add_argument('-dt', '--dt', type=float, default=0.4)
parser.add_argument('-tmax', '--tmax', type=float, default=20.4)
parser.add_argument('-integrator', '--integrator', type=str, default='vi4')
parser.add_argument('-save_name', '--save_name', type=str, default='trials_noise')
parser.add_argument('-num_nodes', '--num_nodes', type=int, default=2)
parser.add_argument('-dname', '--dname', type=str, default='n_grav')
parser.add_argument('-noise_std', '--noise', type=float, default=0)
parser.add_argument('-fname', '--fname', type=str, default='b')
verbose = True
verbose1 = False
args = parser.parse_args()
num_nodes = args.num_nodes
iters = args.num_iters
n_test_traj = args.ntesttraj
num_trajectories = args.ntraintraj
T_max = args.tmax
dt = args.dt
srate = args.srate
# -1 due to down sampling

num_samples_per_traj = int(np.ceil((T_max / dt) / (srate / dt))) - 1
integ = args.integrator
if args.noise != 0:
    noisy = True
else:
    noisy = False
dataset_name = args.dname
expt_name = args.save_name
fname = args.fname

T_max_t = 3*T_max
num_samples_test = int(np.ceil((T_max_t / dt) / (srate / dt))) - 1

# dataset preprocessing
train_data = get_dataset(dataset_name, expt_name, num_trajectories, num_nodes, T_max, dt, srate, args.noise, 0)
valid_data = get_dataset(dataset_name, expt_name, n_test_traj, num_nodes, T_max_t, dt, srate, 0, 11)
BS = num_samples_per_traj
BS_test = num_samples_test
# dimension of a single particle, if 1D, spdim is 2
spdim = int(train_data['x'][0].shape[0] / num_nodes)
print_every = 10

hamiltonian_fn = get_hamiltonian(dataset_name)
# model loop settings
model_types = ['classic']

classic_methods = ['dn', 'hnn','pnn']
graph_methods = ['dgn', 'hogn', 'pgn']

lr_stack = [1e-3]
df_all = pd.DataFrame(
    columns=['model', 'model_type', 'sample',
             'train_state_error', 'train_state_std', 'train_energy_error', 'train_energy_std',
             'valid_state_error', 'valid_std', 'valid_energy_error', 'valid_energy_std',
             'test_state_error', 'test_std', 'test_energy_error', 'test_energy_std'])

for model_type in model_types:
    if model_type == 'classic':
        xnow, xnext, dxnow = nownext(train_data, num_trajectories, num_nodes, T_max, dt, srate,
                                     spatial_dim=spdim, nograph=True)
        test_xnow, test_xnext, test_dxnow = nownext(valid_data, n_test_traj, num_nodes, T_max_t, dt, srate,
                                                    spatial_dim=spdim, nograph=True)

        # tot_train_samples = int(xnow.shape[0])
        #
        # tot_train_samples_valid = int(test_xnow.shape[0])
        #
        # Tot_iters = int(tot_train_samples / (BS))
        # num_training_iterations = int(iters / Tot_iters)

        error_collector = np.zeros((len(lr_stack), len(classic_methods), n_test_traj))
        for lr_index, sublr in enumerate(lr_stack):
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

                gm = nongraph_model(classic_method, num_nodes, BS, integ, expt_name, sublr, noisy, spdim, srate)
                # print(gm.parameters())
                optim = torch.optim.Adam(gm.parameters(), 1e-3,weight_decay=1e-5)
                xvec = np.arange(0, int(num_samples_per_traj*num_trajectories), 1, dtype=int)
                # xvec_valid = np.arange(0, tot_train_samples_valid, 1, dtype=int)
                xnow = torch.tensor(xnow,requires_grad=True,dtype=torch.float32)
                test_xnow = torch.tensor(test_xnow,requires_grad=True,dtype=torch.float32)
                xnext = torch.tensor(xnext, dtype=torch.float32)
                test_xnext = torch.tensor(test_xnext, dtype=torch.float32)

                for iteration in range(iters):
                    for sub_iter in range(num_trajectories):
                        # gm.train()
                        # np.random.shuffle(xvec)
                        # np.random.shuffle(xvec_valid)
                        indices = xvec[sub_iter*BS:(sub_iter+1)*BS]
                        input_batch = xnow[indices[0]]
                        # print(input_batch)
                        true_batch = xnext[indices]
                        loss, _ = gm.train_step(input_batch, true_batch,BS)
                        loss.backward();
                        optim.step();
                        optim.zero_grad()

                        # writer.add_scalar('train_loss', loss.item(), iteration * Tot_iters + sub_iter)
                        if ((iteration) % print_every == 0):
                            print('Iteration:{},Training Loss:{:.3g}'.format(iteration, loss.item()))
                        #     indices_valid = xvec_valid[:BS]
                        #     input_batch = test_xnow[indices_valid]
                        #     true_batch = test_xnext[indices_valid]
                        #     # t1_start = process_time()
                        #     gm.eval()
                        #     loss, _ = gm.valid_step(input_batch, true_batch)
                        #     print('Iteration:{},Validation Loss:{:.3g}'.format(iteration * Tot_iters + sub_iter, loss.item()))
                        #     writer.add_scalar('valid_loss', loss.item(), iteration * Tot_iters + sub_iter)

                gm.eval()
                train_loss, train_pred_state = 1,1#,gm.valid_step(xnow, xnext)
                train_std = 1#((train_pred_state - xnext) ** 2).std()
                hp = 1#hamiltonian_fn(train_pred_state.detach().numpy(), model_type)
                hp_gt = 1#hamiltonian_fn(xnext.detach().numpy(), model_type)
                train_energy_error =1#mean_squared_error(np.sum(hp, 0), np.sum(hp_gt, 0))
                train_energy_std = 1#((np.sum(hp, 0) - np.sum(hp_gt, 0)) ** 2).std()

                valid_loss, valid_pred_state =1,1# gm.valid_step(test_xnow, test_xnext)
                valid_std =1# ((valid_pred_state - test_xnext) ** 2).std()
                hp =1# hamiltonian_fn(valid_pred_state.detach().numpy(), model_type)
                hp_gt = 1#hamiltonian_fn(test_xnext.detach().numpy(), model_type)
                valid_energy_error = 1#mean_squared_error(np.sum(hp, 0), np.sum(hp_gt, 0))
                valid_energy_std = 1#((np.sum(hp, 0) - np.sum(hp_gt, 0)) ** 2).std()

                # print('Iteration:{},Training Loss:{:.3g}'.format(iteration * Tot_iters + sub_iter, loss))


                for t_iters in range(n_test_traj):
                    input_batch = test_xnow[t_iters * BS_test:t_iters * BS_test + 1]
                    true_batch = test_xnext[t_iters * BS_test: (t_iters + 1) * BS_test]
                    error, yhat = gm.test_step(input_batch, true_batch, BS_test)
                    # error_collector[lr_index, gm_index, t_iters] = error
                    hp = hamiltonian_fn(yhat, model_type)
                    hp_gt = hamiltonian_fn(true_batch.detach().numpy(), model_type)
                    state_error = mean_squared_error(yhat, true_batch.detach().numpy())
                    energy_error = mean_squared_error(np.sum(hp, 0), np.sum(hp_gt, 0))
                    test_std = ((yhat - true_batch.detach().numpy()) ** 2).std()
                    test_energy_std = ((np.sum(hp, 0) - np.sum(hp_gt, 0)) ** 2).std()
                    df_all.loc[len(df_all)] = [classic_method, model_type, t_iters,
                                               train_loss, train_std, train_energy_error, train_energy_std,
                                               valid_loss, valid_std, valid_energy_error, valid_energy_std,
                                               state_error, test_std, energy_error, test_energy_std]
                    df_all.to_csv(f'run_data_{dataset_name}_{integ}_{noisy}_{fname}.csv')


    df_all.to_csv(f'run_data_{dataset_name}_{integ}_{noisy}_{fname}.csv')
