"""
Author: ****
code to build graph based models for VIGN

Some aspects adopted from: https://github.com/steindoringi/Variational_Integrator_Networks/blob/master/models.py
"""
from utils import *
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import torch


def create_loss_ops(true, predicted):
    """MSE loss"""
    loss_ops = tf.reduce_mean((true - predicted) ** 2)
    return loss_ops


def log_likelihood_y(y, y_rec, log_noise_var):
    """ noise loss"""
    noise_var = tf.nn.softplus(log_noise_var) * tf.ones_like(y_rec)
    py = tfd.Normal(y_rec, noise_var)
    log_py = py.log_prob(y)
    log_py = tf.reduce_sum(log_py, [0])
    log_lik = tf.reduce_mean(log_py)
    return log_lik


def choose_integrator(method):
    """
    returns integrator for dgn/hnn from utils
    args:
     method (str): 'rk1' or 'rk4'
    """
    if method == 'rk1':
        return rk1

    elif method == 'rk2':
        return rk2

    elif method == 'rk3':
        return rk3

    elif method == 'rk4':
        return rk4

    elif method == 'vi1':
        return vi1

    elif method == 'vi2':
        return vi2

    elif method == 'vi3':
        return vi3

    elif method == 'vi4':
        return vi4


def choose_integrator_nongraph(method):
    """
    returns integrator for dgn/hnn from utils
    args:
     method (str): 'rk1' or 'rk4'
    """
    if method == 'rk1':
        return rk1ng
    elif method == 'rk2':
        return rk2ng
    elif method == 'rk3':
        return rk3ng
    elif method == 'rk4':
        return rk4ng
    elif method == 'vi1':
        return vi1ng
    elif method == 'vi2':
        return vi2ng
    elif method == 'vi3':
        return vi3ng
    elif method == 'vi4':
        return vi4ng


class nongraph_model(torch.nn.Module):

    def __init__(self, deriv_method, num_nodes, BS, integ_meth, expt_name, lr,
                 noisy, spatial_dim, dt):
        """
            Builds a tensorflow classic non-graph model object
            Args:
                sess (tf.session): instantiated session
                deriv_method (str): one of hnn,dgn,vin_rk1,vin_rk4,vin_rk1_lr,vin_rk4_lr
                num_nodes (int): number of particles
                BS (int): batch size
                integ_method (str): rk1 or rk4 for now, though vign has higher order integrators
                expt_name (str): identifier for specific experiment
                lr (float): learning rate
                is_noisy (bool): flag for noisy data
                spatial_dim (int): the dimension of state vector for 1 particle (e.g. 2, [q,qdot] in spring system)
                dt (float): sampling rate
            """
        super(nongraph_model, self).__init__()
        self.deriv_method = deriv_method
        self.num_nodes = num_nodes
        self.BS = BS
        self.BS_test = 1
        self.integ_method = integ_meth
        self.expt_name = expt_name
        self.lr = lr
        self.spatial_dim = spatial_dim
        self.dt = dt
        self.log_noise_var = None
        self.M = self.permutation_tensor(int(self.spatial_dim*self.num_nodes))
        self.hidden_dim = 200
        self._build_net()

    def _build_net(self):
        """
        initializes all placeholders/networks/losses
        """
        if self.deriv_method == 'dn':
            self.input_dim = int(self.spatial_dim * self.num_nodes)
            self.linear1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.linear2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            self.linear3 = torch.nn.Linear(self.hidden_dim, self.input_dim, bias=None)

        elif self.deriv_method == 'hnn':
            self.input_dim = int(self.spatial_dim * self.num_nodes)
            self.linear1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.linear2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            self.linear3 = torch.nn.Linear(self.hidden_dim, 1, bias=None)

        elif self.deriv_method == 'pnn':
            self.input_dim = int(self.spatial_dim * self.num_nodes / 2)
            self.linear1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.linear2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            self.linear3 = torch.nn.Linear(self.hidden_dim, 1, bias=None)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.nonlinearity = torch.nn.Softplus()

        self.mlp = torch.nn.Sequential(
            self.linear1,
            self.nonlinearity,
            self.linear2,
            self.nonlinearity,
            self.linear3
        )

        self.integ = choose_integrator_nongraph(self.integ_method)

        if self.deriv_method == 'dn':
            self.deriv_function = self.deriv_fun_dn
        elif self.deriv_method == 'hnn':
            self.deriv_function = self.deriv_fun_hnn
        else:
            self.deriv_function = self.deriv_fun_pnn

    def deriv_fun_dn(self, xt):
        output_nodes = self.mlp(xt)
        return output_nodes

    def deriv_fun_hnn(self, xt):
        output_nodes = self.mlp(xt)
        dF2 = torch.autograd.grad(output_nodes.sum(), xt, create_graph=True)[0]  # gradients for solenoidal field
        solenoidal_field = dF2 @ self.M.t()
        return solenoidal_field

    def deriv_fun_pnn(self, xt):
        vdim = int(xt.shape[1]/2)
        q, p = xt[:, :vdim], xt[:, vdim:]
        output_nodes = self.mlp(q)
        dF2 = torch.autograd.grad(output_nodes.sum(), q, create_graph=True)[0]  # gradients for solenoidal field
        newdF2 = torch.cat([dF2, p], 1)
        solenoidal_field = newdF2 @ self.M.t()
        return solenoidal_field


    def permutation_tensor(self, n):
        M = None
        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]], 0)
        return M

    def train_step(self, input_batch, true_batch):
        next_step = self.integ(self.deriv_function, input_batch, self.dt)
        return torch.mean(torch.square(next_step- true_batch)), next_step

    def valid_step(self, input_batch, true_batch):
        next_step = self.integ(self.deriv_function, input_batch, self.dt)
        return torch.mean(torch.square(next_step- true_batch)), next_step


    def test_step(self, input_batch, true_batch, steps):
        # figures relegated to jupyter notebook infengine
        stored_states = [input_batch]
        for i in range(steps):
            yhat = self.integ(self.deriv_function,stored_states[-1],self.dt)
            stored_states.append(yhat)

        preds = torch.cat(stored_states).detach().numpy()

        error = mean_squared_error(preds[1:, :], true_batch[:, :].detach().numpy())

        return error, preds[1:, :]
