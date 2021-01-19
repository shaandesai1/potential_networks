"""
Author: ***
Code to produce the results obtained in VIGN: Variational Integrator Graph Networks

"""

from data_builder import *
from models import *
from utils import *
from tensorboardX import SummaryWriter
import argparse
from time import process_time

parser = argparse.ArgumentParser()
parser.add_argument('-ni', '--num_iters', type=int, default=20000)
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
parser.add_argument('-fname', '--fname', type=str, default='a')

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
# dataset preprocessing
train_data = get_dataset(dataset_name, expt_name, num_trajectories, num_nodes, T_max, dt, srate, args.noise, 0)
valid_data = get_dataset(dataset_name, expt_name, n_test_traj, num_nodes, T_max, dt, srate, 0, 11)
BS = 200
BS_test = num_samples_per_traj
# dimension of a single particle, if 1D, spdim is 2
spdim = int(train_data['x'][0].shape[0] / num_nodes)
print_every = 1000

# model loop settings
model_types = ['classic','graphic']

classic_methods = ['dn','hnn','pnn']
graph_methods = ['dgn','hogn','pgn']

lr_stack = [1e-3]
for model_type in model_types:
    if model_type == 'classic':
        xnow, xnext, dxnow = nownext(train_data, num_trajectories, num_nodes, T_max, dt, srate,
                                     spatial_dim=spdim,nograph=True)
        test_xnow, test_xnext, test_dxnow = nownext(valid_data, n_test_traj, num_nodes, T_max, dt, srate,
                                                    spatial_dim=spdim,nograph=True)



        tot_train_samples = int(xnow.shape[0])

        tot_train_samples_valid = int(test_xnow.shape[0])

        Tot_iters = int(tot_train_samples / (BS))
        num_training_iterations = int(iters / Tot_iters)

        error_collector = np.zeros((len(lr_stack), len(classic_methods), n_test_traj))
        for lr_index, sublr in enumerate(lr_stack):
            for gm_index, classic_method in enumerate(classic_methods):

                #pass the masses to the function if we have them
                #mainly used for simultaneously learning from systems with different masses during training
                #can abstract this and allow the network
                # if classic_method == 'pnn':
                #     newmass = np.repeat(train_data['mass'], num_samples_per_traj, axis=0)
                #     subdim_ = int(spdim / 2)
                #     if subdim_ != 1:
                #         newmass = np.repeat(newmass, subdim_, axis=1)
                #     xnow[:,int(subdim_*num_nodes):] = xnow[:,int(subdim_*num_nodes):]/newmass
                #     xnext[:, int(subdim_ * num_nodes):] = xnext[:, int(subdim_ * num_nodes):] / newmass
                #     test_xnow[:, int(subdim_ * num_nodes):] = test_xnow[:, int(subdim_ * num_nodes):] / newmass
                #     test_xnext[:, int(subdim_ * num_nodes):] = test_xnext[:, int(subdim_ * num_nodes):] / newmass

                data_dir = 'data/' + dataset_name + '/' + str(sublr) + '/' + classic_method + '/'+ fname + '/'
                if not os.path.exists(data_dir):
                    print('non existent path....creating path')
                    os.makedirs(data_dir)
                dirw = classic_method + str(sublr) + integ + fname
                if noisy:
                    writer = SummaryWriter('noisy/' + dataset_name + '/' + str(sublr) + '/' + classic_method + '/' + dirw)
                else:
                    writer = SummaryWriter(
                        'noiseless/' + dataset_name + '/' + str(sublr) + '/' + classic_method + '/' + dirw)
                try:
                    sess.close()
                except NameError:
                    pass

                tf.reset_default_graph()
                sess = tf.Session()
                gm = nongraph_model(sess,classic_method, num_nodes, BS, integ, expt_name, sublr, noisy, spdim, srate)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                xvec = np.arange(0, tot_train_samples, 1, dtype=int)
                xvec_valid = np.arange(0, tot_train_samples_valid, 1, dtype=int)
                for iteration in range(num_training_iterations):
                    np.random.shuffle(xvec)
                    np.random.shuffle(xvec_valid)
                    for sub_iter in range(Tot_iters):
                        input_batch = np.vstack(
                            [xnow[xvec[i]] for i in
                             range(sub_iter * BS, (sub_iter + 1) * BS)])
                        true_batch = np.vstack(
                            [xnext[xvec[i]] for i in
                             range(sub_iter * BS, (sub_iter + 1) * BS)])
                        # batch_masses = np.vstack([newmass[xvec[i]]] for i in range(sub_iter*BS,(sub_iter+1)*BS))
                        loss = gm.train_step(input_batch, true_batch)
                        # t1_end = process_time()
                        writer.add_scalar('train_loss', loss, iteration * Tot_iters + sub_iter)
                        if ((iteration * Tot_iters + sub_iter) % print_every == 0):
                            print('Iteration:{},Training Loss:{:.3g}'.format(iteration * Tot_iters + sub_iter, loss))
                            input_batch = np.vstack(
                                [test_xnow[xvec_valid[i]] for i in
                                 range(0 * BS, (0 + 1) * BS)])
                            true_batch = np.vstack(
                                [test_xnext[xvec_valid[i]] for i in
                                 range(0 * BS, (0 + 1) * BS)])
                            # t1_start = process_time()
                            loss = gm.valid_step(input_batch, true_batch)
                            print('Iteration:{},Validation Loss:{:.3g}'.format(iteration * Tot_iters + sub_iter, loss))
                            writer.add_scalar('valid_loss', loss, iteration * Tot_iters + sub_iter)

                            # print('Time:{}'.format(t1_end - t1_start))
                            # saves model every 1000 iters (I/O slow)
                            # if noisy:
                            #     saver.save(sess, data_dir + graph_method + str(sublr) + integ + 'noisy')
                            # else:
                            #     saver.save(sess, data_dir + graph_method + str(sublr) + integ)

                print('Iteration:{},Training Loss:{:.3g}'.format(iteration * Tot_iters + sub_iter, loss))

                if noisy:
                    saver.save(sess, data_dir + classic_method + str(sublr) + integ + fname + 'noisy')
                else:
                    saver.save(sess, data_dir + classic_method + str(sublr) + integ + fname)

                # for t_iters in range(n_test_traj):
                #     input_batch = test_xnow[t_iters*BS_test:t_iters*BS_test+1]
                #     true_batch = test_xnext[t_iters * BS_test: (t_iters + 1) * BS_test]
                #     error, _ = gm.test_step(input_batch, true_batch, BS_test)
                #     error_collector[lr_index, gm_index, t_iters] = error
                # print('mean test error:{}'.format(error_collector[lr_index, :, :].mean(1)))
                # print('std test error:{}'.format(error_collector[lr_index, :, :].std(1)))
            if noisy:
                np.save(data_dir +dirw+ 'classic_collater_noisy' + '.npy', error_collector)
            else:
                np.save(data_dir +dirw+'classic_collater' + '.npy', error_collector)


    elif model_type == 'graphic':
        xnow, xnext, dxnow = nownext(train_data, num_trajectories, num_nodes, T_max, dt, srate,
                                     spatial_dim=spdim)
        newmass = np.repeat(train_data['mass'], num_samples_per_traj, axis=0)
        newks = np.repeat(train_data['ks'], num_samples_per_traj, axis=0)
        test_xnow, test_xnext, test_dxnow = nownext(valid_data, n_test_traj, num_nodes, T_max, dt, srate,
                                                    spatial_dim=spdim)
        test_mass = np.repeat(valid_data['mass'], num_samples_per_traj, axis=0)
        test_ks = np.repeat(valid_data['ks'], num_samples_per_traj, axis=0)
        tot_train_samples = int(xnow.shape[0] / num_nodes)

        tot_train_samples_valid = int(test_xnow.shape[0] / num_nodes)

        Tot_iters = int(tot_train_samples / (BS))
        num_training_iterations = int(iters / Tot_iters)

        error_collector = np.zeros((len(lr_stack), len(graph_methods), n_test_traj))
        for lr_index, sublr in enumerate(lr_stack):
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
                gm = graph_model(sess, graph_method, num_nodes, BS, integ, expt_name, sublr, noisy, spdim, srate, True)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                xvec = np.arange(0, tot_train_samples, 1, dtype=int)
                xvec_valid = np.arange(0, tot_train_samples_valid, 1, dtype=int)
                for iteration in range(num_training_iterations):
                    np.random.shuffle(xvec)
                    np.random.shuffle(xvec_valid)
                    for sub_iter in range(Tot_iters):
                        input_batch = np.vstack(
                            [xnow[xvec[i] * num_nodes:xvec[i] * num_nodes + num_nodes] for i in
                             range(sub_iter * BS, (sub_iter + 1) * BS)])
                        true_batch = np.vstack(
                            [xnext[xvec[i] * num_nodes:xvec[i] * num_nodes + num_nodes] for i in
                             range(sub_iter * BS, (sub_iter + 1) * BS)])
                        ks_true = np.vstack([newks[xvec[i]] for i in range(sub_iter * BS, (sub_iter + 1) * BS)])
                        ms_true = np.vstack([newmass[xvec[i]] for i in range(sub_iter * BS, (sub_iter + 1) * BS)])
                        # t1_start = process_time()
                        loss = gm.train_step(input_batch, true_batch, ks_true, ms_true)
                        # t1_end = process_time()
                        writer.add_scalar('train_loss', loss, iteration * Tot_iters + sub_iter)
                        if ((iteration * Tot_iters + sub_iter) % print_every == 0):
                            print('Iteration:{},Training Loss:{:.3g}'.format(iteration * Tot_iters + sub_iter, loss))
                            input_batch = np.vstack(
                                [test_xnow[xvec_valid[i] * num_nodes:xvec_valid[i] * num_nodes + num_nodes] for i in
                                 range(0 * BS, (0 + 1) * BS)])
                            true_batch = np.vstack(
                                [test_xnext[xvec_valid[i] * num_nodes:xvec_valid[i] * num_nodes + num_nodes] for i in
                                 range(0 * BS, (0 + 1) * BS)])
                            ks_true = np.vstack(
                                [test_ks[xvec_valid[i]] for i in range(0 * BS, (0 + 1) * BS)])
                            ms_true = np.vstack(
                                [test_mass[xvec_valid[i]] for i in range(0 * BS, (0 + 1) * BS)])
                            # t1_start = process_time()
                            loss = gm.valid_step(input_batch, true_batch, ks_true, ms_true)
                            print('Iteration:{},Validation Loss:{:.3g}'.format(iteration * Tot_iters + sub_iter, loss))
                            writer.add_scalar('valid_loss', loss, iteration * Tot_iters + sub_iter)

                            # print('Time:{}'.format(t1_end - t1_start))
                            # saves model every 1000 iters (I/O slow)
                            # if noisy:
                            #     saver.save(sess, data_dir + graph_method + str(sublr) + integ + 'noisy')
                            # else:
                            #     saver.save(sess, data_dir + graph_method + str(sublr) + integ)

                print('Iteration:{},Training Loss:{:.3g}'.format(iteration * Tot_iters + sub_iter, loss))

                if noisy:
                    saver.save(sess, data_dir + graph_method + str(sublr) + integ + fname + 'noisy')
                else:
                    saver.save(sess, data_dir + graph_method + str(sublr) + integ + fname)

                # for t_iters in range(n_test_traj):
                #     input_batch = test_xnow[num_nodes * t_iters * BS_test:num_nodes * t_iters * BS_test + num_nodes]
                #     true_batch = test_xnext[num_nodes * t_iters * BS_test:num_nodes * (t_iters + 1) * BS_test]
                #     error, _ = gm.test_step(input_batch, true_batch, np.reshape(test_ks[t_iters * BS_test], [1, -1]),
                #                             np.reshape(test_mass[t_iters * BS_test], [1, -1]), BS_test)
                #     error_collector[lr_index, gm_index, t_iters] = error
                # print('mean test error:{}'.format(error_collector[lr_index, :, :].mean(1)))
                # print('std test error:{}'.format(error_collector[lr_index, :, :].std(1)))
            if noisy:
                np.save(data_dir + 'graphic_collater_noisy' + '.npy', error_collector)
            else:
                np.save(data_dir + 'graphic_collater' + '.npy', error_collector)

    try:
        sess.close()
    except NameError:
        print('NameError')
