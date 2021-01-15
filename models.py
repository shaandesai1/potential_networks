"""
Author: ****
code to build graph based models for VIGN

Some aspects adopted from: https://github.com/steindoringi/Variational_Integrator_Networks/blob/master/models.py
"""
from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt
import tensorflow as tf
from utils import rk1, rk2, rk3, rk4, rk1ng, rk2ng, rk3ng, rk4ng
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow.keras as tfk
from tensorflow_probability import distributions as tfd
import os


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


class nongraph_model(object):

    def __init__(self, sess, deriv_method, num_nodes, BS, integ_meth, expt_name, lr,
                 noisy, spatial_dim, dt):

        self.sess = sess
        self.deriv_method = deriv_method
        self.num_nodes = num_nodes
        self.BS = BS
        self.BS_test = 1
        self.integ_method = integ_meth
        self.expt_name = expt_name
        self.lr = lr
        self.spatial_dim = spatial_dim
        self.dt = dt
        # self.eflag = eflag
        self.is_noisy = noisy
        self.log_noise_var = None
        if self.num_nodes == 1:
            self.activate_sub = False
        else:
            self.activate_sub = True
        self.output_plots = False
        self.M = tf.transpose(self.permutation_tensor(self.spatial_dim * self.num_nodes))
        self._build_net()

    def _build_net(self):
        """
        initializes all tf placeholders/graph networks/losses
        """

        if self.is_noisy:
            self.log_noise_var = tf.Variable([0.], dtype=tfk.backend.floatx())

        self.nonlin = tf.nn.softplus

        self.h1 = snt.Linear(output_size=200, use_bias=True, name='h1')
        self.h2 = snt.Linear(output_size=200, use_bias=True, name='h2')

        if self.deriv_method == 'dn':
            self.h3 = snt.Linear(output_size=self.spatial_dim * self.num_nodes, use_bias=True, name='h3')
        else:
            self.h3 = snt.Linear(output_size=1, use_bias=False, name='h3')

        self.mlp = snt.Sequential([
            self.h1,
            self.nonlin,
            self.h2,
            self.nonlin,
            self.h3
        ])

        self.input_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS, self.spatial_dim * self.num_nodes])
        self.test_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, self.spatial_dim * self.num_nodes])
        self.ground_truth_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS, self.spatial_dim * self.num_nodes])

        integ = choose_integrator_nongraph(self.integ_method)

        if self.deriv_method == 'dn':
            next_step = integ(self.deriv_fun_dn, self.input_ph, self.dt)
            self.test_next_step = integ(self.deriv_fun_dn, self.test_ph, self.dt)

        elif self.deriv_method == 'hnn':
            next_step = integ(self.deriv_fun_hnn, self.input_ph, self.dt)
            self.test_next_step = integ(self.deriv_fun_hnn, self.test_ph, self.dt)

        elif self.deriv_method == 'pnn':
            next_step = integ(self.deriv_fun_pnn, self.input_ph, self.dt)
            self.test_next_step = integ(self.deriv_fun_pnn, self.test_ph, self.dt)

        else:
            raise ValueError("the derivative generator is incorrect, should be dn,hnn or pn")

        if self.is_noisy:
            self.loss_op_tr = -self.log_likelihood_y(next_step, self.ground_truth_ph, self.log_noise_var)
        else:
            self.loss_op_tr = self.create_loss_ops(next_step, self.ground_truth_ph)

        global_step = tf.compat.v1.Variable(0, trainable=False)
        rate = tf.compat.v1.train.exponential_decay(self.lr, global_step, 10000, 0.5, staircase=False)
        optimizer = tf.compat.v1.train.AdamOptimizer(rate)
        self.step_op = optimizer.minimize(self.loss_op_tr, global_step=global_step)

    def create_loss_ops(self, true, predicted):
        """MSE loss"""
        loss_ops = tf.reduce_mean((true - predicted) ** 2)
        return loss_ops

    def log_likelihood_y(self, y, y_rec, log_noise_var):
        """ noise loss"""
        noise_var = tf.nn.softplus(log_noise_var) * tf.ones_like(y_rec)
        py = tfd.Normal(y_rec, noise_var)
        log_py = py.log_prob(y)
        log_py = tf.reduce_sum(log_py, [0])
        log_lik = tf.reduce_mean(log_py)
        return log_lik

    def deriv_fun_dn(self, xt):
        output_nodes = self.mlp(xt)
        return output_nodes

    def deriv_fun_hnn(self, xt):
        with tf.GradientTape() as g:
            g.watch(xt)
            output_nodes = self.mlp(xt)
        dH = g.gradient(output_nodes, xt)
        return tf.matmul(dH, self.M)

    def deriv_fun_pnn(self, xt):
        qvals = xt[:, :int(self.spatial_dim )]
        pvals = xt[:, int(self.spatial_dim ):]
        with tf.GradientTape() as g:
            g.watch(qvals)
            output_nodes = self.mlp(qvals)
        dH = g.gradient(output_nodes, qvals)
        return tf.concat([pvals, -dH], 1)

    def permutation_tensor(self, n):
        M = None
        M = tf.eye(n)
        M = tf.concat([M[n // 2:], -M[:n // 2]],0)
        return M

    def train_step(self, input_batch, true_batch):

        train_feed = {self.input_ph: input_batch,
                      self.ground_truth_ph: true_batch,
                      }
        train_ops = [self.loss_op_tr, self.step_op]
        loss, _ = self.sess.run(train_ops, feed_dict=train_feed)
        return loss

    def valid_step(self, input_batch, true_batch):

        train_feed = {self.input_ph: input_batch,
                      self.ground_truth_ph: true_batch,
                      }
        train_ops = self.loss_op_tr
        loss = self.sess.run(train_ops, feed_dict=train_feed)

        return loss

    def test_step(self, input_batch, true_batch, steps):
        # figures relegated to jupyter notebook infengine
        stored_states = [input_batch.astype(np.float32)]
        for i in range(steps):
            test_feed = {self.test_ph: stored_states[-1],
                         }
            test_ops = [self.test_next_step]

            yhat = self.sess.run(test_ops, feed_dict=test_feed)
            stored_states.append(yhat[0])

        preds = tf.concat(stored_states, 0).eval(session=self.sess)

        error = mean_squared_error(preds[1:, :], true_batch[:, :])

        return error, preds


class graph_model(object):
    """
    Builds a tensorflow graph model object
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
        eflag (bool): whether to use extra input in building graph (default=True)
    """

    def __init__(self, sess, deriv_method, num_nodes, BS, integ_meth, expt_name, lr,
                 noisy, spatial_dim, dt, eflag=True):

        self.sess = sess
        self.deriv_method = deriv_method
        self.num_nodes = num_nodes
        self.BS = BS
        self.BS_test = 1
        self.integ_method = integ_meth
        self.expt_name = expt_name
        self.lr = lr
        self.spatial_dim = spatial_dim
        self.dt = dt
        self.eflag = eflag
        self.is_noisy = noisy
        self.log_noise_var = None
        if self.num_nodes == 1:
            self.activate_sub = False
        else:
            self.activate_sub = True
        self.output_plots = False
        self._build_net()

    def _build_net(self):
        """
        initializes all tf placeholders/graph networks/losses
        """

        if self.is_noisy:
            self.log_noise_var = tf.Variable([0.], dtype=tfk.backend.floatx())

        self.out_to_global = snt.Linear(output_size=1, use_bias=False, name='out_to_global')
        self.out_to_node = snt.Linear(output_size=self.spatial_dim, use_bias=True, name='out_to_node')

        self.graph_network = modules.GraphNetwork(
            edge_model_fn=lambda: snt.nets.MLP([32, 32], activation=tf.nn.softplus, activate_final=True),
            node_model_fn=lambda: snt.nets.MLP([32, 32], activation=tf.nn.softplus, activate_final=True),
            global_model_fn=lambda: snt.nets.MLP([32, 32], activation=tf.nn.softplus, activate_final=True),
        )

        self.base_graph_tr = tf.compat.v1.placeholder(tf.float32,
                                                      shape=[self.num_nodes * self.BS, self.spatial_dim])
        self.ks_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS, self.num_nodes])
        self.ms_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS, self.num_nodes])
        self.true_dq_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, self.spatial_dim])

        self.test_graph_ph = tf.compat.v1.placeholder(tf.float32,
                                                      shape=[self.num_nodes * self.BS_test, self.spatial_dim])
        self.test_ks_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS_test, self.num_nodes])
        self.test_ms_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS_test, self.num_nodes])

        integ = choose_integrator(self.integ_method)

        if self.deriv_method == 'dgn_rk4':
            next_step = integ(self.deriv_fun_dgn, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                              self.num_nodes)
            self.test_next_step = integ(self.deriv_fun_dgn, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                        self.dt, 1, self.num_nodes)

        elif self.deriv_method == 'hnn_rk4':
            next_step = integ(self.deriv_fun_hnn, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                              self.num_nodes)
            self.test_next_step = integ(self.deriv_fun_hnn, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                        self.dt, 1, self.num_nodes)
        elif self.deriv_method == 'hnn_vi2':
            next_step = self.hnn_vi2(self.deriv_fun_hnn, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                                     self.num_nodes)
            self.test_next_step = self.hnn_vi2(self.deriv_fun_hnn, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                               self.dt, 1, self.num_nodes)

        elif self.deriv_method == 'hnn_vi4':
            next_step = self.hnn_vi4(self.deriv_fun_hnn, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                                     self.num_nodes)
            self.test_next_step = self.hnn_vi4(self.deriv_fun_hnn, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                               self.dt, 1, self.num_nodes)

        elif self.deriv_method == "vin_rk2":
            next_step = self.rk2(self.deriv_fun_vin, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                                 self.num_nodes)
            self.test_next_step = self.rk2(self.deriv_fun_vin, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                           self.dt, 1, self.num_nodes)
        elif self.deriv_method == 'vin_rk4':
            next_step = self.rk4_vin(self.deriv_fun_vin, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                                     self.num_nodes)
            self.test_next_step = self.rk4_vin(self.deriv_fun_vin, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                               self.dt, 1, self.num_nodes)

        elif self.deriv_method == "vin_vi2":
            next_step = self.vin2(self.deriv_fun_vin, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                                  self.num_nodes)
            self.test_next_step = self.vin2(self.deriv_fun_vin, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                            self.dt, 1, self.num_nodes)

        elif self.deriv_method == "vin_vi4":
            next_step = self.vin4(self.deriv_fun_vin, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                                  self.num_nodes)
            self.test_next_step = self.vin4(self.deriv_fun_vin, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                            self.dt, 1, self.num_nodes)

        elif self.deriv_method == "vin_vi4_2":
            next_step = self.vin4_2(self.deriv_fun_vin, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                                    self.num_nodes)
            self.test_next_step = self.vin4_2(self.deriv_fun_vin, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                              self.dt, 1, self.num_nodes)


        elif self.deriv_method == 'vin_vi4_lr':
            next_step = self.future_pred(self.vin4, self.deriv_fun_vin, self.base_graph_tr, self.ks_ph, self.ms_ph,
                                         self.dt, 1, self.num_nodes)
            self.test_next_step = self.vin4(self.deriv_fun_vin, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                            self.dt, 1, self.num_nodes)

        if self.is_noisy:
            self.loss_op_tr = -self.log_likelihood_y(next_step, self.true_dq_ph, self.log_noise_var)
        else:
            self.loss_op_tr = self.create_loss_ops(self.true_dq_ph, next_step)

        # if self.deriv_method == 'vin_rk4_lr':
        #     # alternative loss
        #     self.loss_op_tr = tf.reduce_mean(tf.reduce_sum((next_step - self.true_dq_ph) ** 2, 0))

        global_step = tf.compat.v1.Variable(0, trainable=False)
        # if self.deriv_method =='vin_vi4':
        rate = tf.compat.v1.train.exponential_decay(self.lr, global_step, 10000, 0.5, staircase=False)
        # else:
        #    rate = tf.compat.v1.train.exponential_decay(self.lr, global_step, 20000, 0.5, staircase=False)

        optimizer = tf.compat.v1.train.AdamOptimizer(rate)
        self.step_op = optimizer.minimize(self.loss_op_tr, global_step=global_step)

    def future_pred(self, integ, deriv_fun, state_init, ks, ms, dt, bs, num_nodes):
        """
        only used with vign_lr - future step predictions
        """
        accum = []

        q_init = state_init[:num_nodes]
        ks_init = ks[0]
        ms_init = ms[0]
        accum.append(q_init)

        for _ in range(self.BS):
            xtp1 = integ(deriv_fun, accum[-1], ks_init, ms_init, dt, bs, num_nodes)
            accum.append(xtp1)

        yhat = tf.concat(accum, 0)
        return yhat[num_nodes:]

    def create_loss_ops(self, true, predicted):
        """MSE loss"""
        loss_ops = tf.reduce_mean((true - predicted) ** 2)
        return loss_ops

    def base_graph(self, input_features, ks, ms, num_nodes):
        """builds graph for every group of particles"""
        # Node features for graph 0.
        if self.eflag:
            nodes_0 = tf.concat([input_features, tf.reshape(ms, [num_nodes, 1]), tf.reshape(ks, [num_nodes, 1])], 1)
        else:
            nodes_0 = input_features

        senders_0 = []
        receivers_0 = []
        edges_0 = []
        an = np.arange(0, num_nodes, 1)
        for i in range(len(an)):
            for j in range(i + 1, len(an)):
                senders_0.append(i)
                senders_0.append(j)
                receivers_0.append(j)
                receivers_0.append(i)

        data_dict_0 = {
            "nodes": nodes_0,
            "senders": senders_0,
            "receivers": receivers_0
        }

        return data_dict_0

    def rk2(self, dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
        """classical vign in newtonian mechanical system"""
        m_init = tf.reshape(ms, [-1, 1])
        m_init = tf.repeat(m_init, int(self.spatial_dim / 2), axis=1)
        init_x, init_v = x_t[:, :int(self.spatial_dim / 2)], x_t[:, int(self.spatial_dim / 2):] / m_init

        vddot = -dx_dt_fn(init_x, ks, ms, bs, nodes) / m_init
        x1 = init_x + init_v * self.dt / 2
        v1 = init_v + vddot * self.dt / 2

        vddot1 = -dx_dt_fn(x1, ks, ms, bs, nodes) / m_init
        x2 = init_x + v1 * self.dt
        v2 = init_v + vddot1 * self.dt

        fin_p = m_init * v2
        return tf.concat([x2, fin_p], 1)

    def rk1_vin(self, dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
        """classical vign in newtonian mechanical system"""
        m_init = tf.reshape(ms, [-1, 1])
        m_init = tf.repeat(m_init, int(self.spatial_dim / 2), axis=1)
        init_x, init_v = x_t[:, :int(self.spatial_dim / 2)], x_t[:, int(self.spatial_dim / 2):] / m_init

        k0 = dt * init_v
        l0 = -dt * dx_dt_fn(init_x, ks, ms, bs, nodes) / m_init

        fin_x = init_x + k0
        fin_v = init_v + l0

        fin_p = m_init * fin_v
        return tf.concat([fin_x, fin_p], 1)

    def hnn_vi2(self, dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
        """classical vign in newtonian mechanical system"""
        # m_init = tf.reshape(ms, [-1, 1])
        # m_init = tf.repeat(m_init, int(self.spatial_dim / 2), axis=1)
        # init_x, init_v = x_t[:, :int(self.spatial_dim / 2)], x_t[:, int(self.spatial_dim / 2):] / m_init

        p = x_t[:, int(self.spatial_dim / 2):]
        q = x_t[:, :int(self.spatial_dim / 2)]

        phalf = p + (self.dt / 2) * dx_dt_fn(x_t, ks, ms, bs, nodes)[:, int(self.spatial_dim / 2):]
        qnext = q + (self.dt / 2) * 2 * dx_dt_fn(tf.concat([q, phalf], 1), ks, ms, bs, nodes)[:,
                                        :int(self.spatial_dim / 2)]
        pnext = phalf + (self.dt / 2) * dx_dt_fn(tf.concat([qnext, phalf], 1), ks, ms, bs, nodes)[:,
                                        int(self.spatial_dim / 2):]

        return tf.concat([qnext, pnext], 1)

    def vin2(self, dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
        """classical vign in newtonian mechanical system"""
        m_init = tf.reshape(ms, [-1, 1])
        m_init = tf.repeat(m_init, int(self.spatial_dim / 2), axis=1)
        init_x, init_v = x_t[:, :int(self.spatial_dim / 2)], x_t[:, int(self.spatial_dim / 2):] / m_init

        q = init_x
        p = init_v
        dUdq = -dx_dt_fn(q, ks, ms, bs, nodes) / m_init
        qddot = dUdq
        q_next = q + dt * p + 0.5 * (dt ** 2) * qddot
        dUdq_next = -dx_dt_fn(q_next, ks, ms, bs, nodes) / m_init
        dUdq_mid = dUdq + dUdq_next
        qddot_mid = dUdq_mid
        p_next = p + 0.5 * self.dt * qddot_mid
        fin_x = q_next
        fin_p = m_init * p_next
        return tf.concat([fin_x, fin_p], 1)

    def hnn_vi4(self, dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
        """classical vign in newtonian mechanical system"""
        m_init = tf.reshape(ms, [-1, 1])
        m_init = tf.repeat(m_init, int(self.spatial_dim / 2), axis=1)
        init_x, init_v = x_t[:, :int(self.spatial_dim / 2)], x_t[:, int(self.spatial_dim / 2):] / m_init

        w0 = -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3))
        w1 = 1 / (2 - 2 ** (1 / 3))
        c1 = c4 = w1 / 2
        c2 = c3 = (w0 + w1) / 2
        d1 = d3 = w1
        d2 = w0

        q = init_x
        p = init_v

        q1 = q + c1 * dt * dx_dt_fn(tf.concat([q, p], 1), ks, ms, bs, nodes)[:, :int(self.spatial_dim / 2)]
        p1 = p + dt * d1 * dx_dt_fn(tf.concat([q1, p], 1), ks, ms, bs, nodes)[:, int(self.spatial_dim / 2):]
        q2 = q1 + c2 * dt * dx_dt_fn(tf.concat([q1, p1], 1), ks, ms, bs, nodes)[:, :int(self.spatial_dim / 2)]
        p2 = p1 + dt * d2 * dx_dt_fn(tf.concat([q2, p1], 1), ks, ms, bs, nodes)[:, int(self.spatial_dim / 2):]
        q3 = q2 + c3 * dt * dx_dt_fn(tf.concat([q2, p2], 1), ks, ms, bs, nodes)[:, :int(self.spatial_dim / 2)]
        p3 = p2 + dt * d3 * dx_dt_fn(tf.concat([q3, p2], 1), ks, ms, bs, nodes)[:, int(self.spatial_dim / 2):]

        q_next = q3 + c4 * dt * p3
        p_next = m_init * p3

        return tf.concat([q_next, p_next], 1)

    def vin4(self, dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
        """classical vign in newtonian mechanical system"""
        m_init = tf.reshape(ms, [-1, 1])
        m_init = tf.repeat(m_init, int(self.spatial_dim / 2), axis=1)
        init_x, init_v = x_t[:, :int(self.spatial_dim / 2)], x_t[:, int(self.spatial_dim / 2):] / m_init

        w0 = -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3))
        w1 = 1 / (2 - 2 ** (1 / 3))
        c1 = c4 = w1 / 2
        c2 = c3 = (w0 + w1) / 2
        d1 = d3 = w1
        d2 = w0

        q = init_x
        p = init_v

        q1 = q + c1 * dt * p
        p1 = p + dt * d1 * -dx_dt_fn(q1, ks, ms, bs, nodes) / m_init
        q2 = q1 + c2 * dt * p1
        p2 = p1 + dt * d2 * -dx_dt_fn(q2, ks, ms, bs, nodes) / m_init
        q3 = q2 + c3 * dt * p2
        p3 = p2 + dt * d3 * -dx_dt_fn(q3, ks, ms, bs, nodes) / m_init

        q_next = q3 + c4 * dt * p3
        p_next = m_init * p3

        return tf.concat([q_next, p_next], 1)

    def vin4_2(self, dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
        """classical vign in newtonian mechanical system"""
        m_init = tf.reshape(ms, [-1, 1])
        m_init = tf.repeat(m_init, int(self.spatial_dim / 2), axis=1)
        init_x, init_v = x_t[:, :int(self.spatial_dim / 2)], x_t[:, int(self.spatial_dim / 2):] / m_init

        a1 = a4 = 1. / 6 * (2 + 2 ** (1 / 3) + 2 ** (-1 / 3))
        a2 = a3 = 1. / 6 * (1 - 2 ** (1 / 3) - 2 ** (-1 / 3))
        b1 = 0
        b2 = b4 = 1. / (2 - 2 ** (1 / 3))
        b3 = 1. / (1 - 2 ** (2 / 3))

        q = init_x
        p = init_v

        p1 = p + dt * b1 * -dx_dt_fn(q, ks, ms, bs, nodes) / m_init  # (dx_dt_fn(np.array([q1,p]),t))[1]
        q1 = q + a1 * dt * p1
        p2 = p1 + dt * b2 * -dx_dt_fn(q1, ks, ms, bs, nodes) / m_init  # (dx_dt_fn(np.array([q2,p1]),t))[1]
        q2 = q1 + a2 * dt * p2
        p3 = p2 + dt * b3 * -dx_dt_fn(q2, ks, ms, bs, nodes) / m_init  # (dx_dt_fn(np.array([q3,p2]),t))[1]
        q3 = q2 + a3 * dt * p3
        p4 = p3 + dt * b4 * -dx_dt_fn(q3, ks, ms, bs, nodes) / m_init
        q4 = q3 + a4 * dt * p4

        return tf.concat([q4, p4], 1)

    def rk4_vin(self, dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
        m_init = tf.reshape(ms, [-1, 1])
        m_init = tf.repeat(m_init, int(self.spatial_dim / 2), axis=1)
        init_x, init_v = x_t[:, :int(self.spatial_dim / 2)], x_t[:, int(self.spatial_dim / 2):] / m_init

        k0 = dt * init_v
        l0 = -dt * dx_dt_fn(init_x, ks, ms, bs, nodes) / m_init
        k1 = dt * (init_v + 0.5 * l0)
        l1 = -dt * dx_dt_fn(init_x + 0.5 * k0, ks, ms, bs, nodes) / m_init
        k2 = dt * (init_v + 0.5 * l1)
        l2 = -dt * dx_dt_fn(init_x + 0.5 * k1, ks, ms, bs, nodes) / m_init
        k3 = dt * (init_v + l2)
        l3 = -dt * dx_dt_fn(init_x + k2, ks, ms, bs, nodes) / m_init
        fin_x = init_x + (1. / 6) * (k0 + 2. * k1 + 2. * k2 + k3)
        fin_v = init_v + (1. / 6) * (l0 + 2. * l1 + 2. * l2 + l3)

        fin_p = m_init * fin_v
        return tf.concat([fin_x, fin_p], 1)

    def deriv_fun_hnn(self, xt, ks, ms, bs, n_nodes):
        if self.activate_sub == True:
            sub_vecs = [self.sub_mean(xt[n_nodes * i:n_nodes * (i + 1), :]) for i in range(bs)]
        else:
            sub_vecs = [xt[n_nodes * i:n_nodes * (i + 1), :] for i in range(bs)]

        input_vec = tf.concat(sub_vecs, 0)
        with tf.GradientTape() as g:
            g.watch(input_vec)
            vec2g = [self.base_graph(input_vec[n_nodes * i:n_nodes * (i + 1)], ks[i], ms[i], n_nodes) for i in
                     range(bs)]
            vec2g = utils_tf.data_dicts_to_graphs_tuple(vec2g)
            vec2g = utils_tf.set_zero_global_features(vec2g, 1)
            vec2g = utils_tf.set_zero_edge_features(vec2g, 1)
            output_graphs = self.graph_network(vec2g)
            global_vals = self.out_to_global(output_graphs.globals)
        dUdq = g.gradient(global_vals, input_vec)

        dqdt = dUdq[:, int(self.spatial_dim / 2):]
        dpdt = -dUdq[:, :int(self.spatial_dim / 2)]
        dHdin = tf.concat([dqdt, dpdt], 1)
        return dHdin

    def deriv_fun_dgn(self, xt, ks, ms, bs, n_nodes):
        if self.activate_sub == True:
            sub_vecs = [self.sub_mean(xt[n_nodes * i:n_nodes * (i + 1), :]) for i in range(bs)]
        else:
            sub_vecs = [xt[n_nodes * i:n_nodes * (i + 1), :] for i in range(bs)]

        input_vec = tf.concat(sub_vecs, 0)
        vec2g = [self.base_graph(input_vec[n_nodes * i:n_nodes * (i + 1)], ks[i], ms[i], n_nodes) for i in range(bs)]
        vec2g = utils_tf.data_dicts_to_graphs_tuple(vec2g)
        vec2g = utils_tf.set_zero_global_features(vec2g, 1)
        vec2g = utils_tf.set_zero_edge_features(vec2g, 1)
        output_graphs = self.graph_network(vec2g)
        new_node_vals = self.out_to_node(output_graphs.nodes)
        return new_node_vals

    def log_likelihood_y(self, y, y_rec, log_noise_var):
        """ noise loss"""
        noise_var = tf.nn.softplus(log_noise_var) * tf.ones_like(y_rec)
        py = tfd.Normal(y_rec, noise_var)
        log_py = py.log_prob(y)
        log_py = tf.reduce_sum(log_py, [0])
        log_lik = tf.reduce_mean(log_py)
        return log_lik

    def sub_mean(self, xt):
        init_x = xt[:, :int(self.spatial_dim / 2)]
        means = tf.reduce_mean(init_x, 0)
        new_means = tf.transpose(tf.reshape(tf.repeat(means, init_x.shape[0]), (int(self.spatial_dim / 2), -1)))
        return tf.concat([init_x - new_means, xt[:, int(self.spatial_dim / 2):]], 1)

    def deriv_fun_vin(self, xt, ks, ms, bs, n_nodes):
        if self.activate_sub == True:
            sub_vecs = [self.sub_mean(xt[n_nodes * i:n_nodes * (i + 1), :]) for i in range(bs)]
        else:
            sub_vecs = [xt[n_nodes * i:n_nodes * (i + 1), :] for i in range(bs)]

        input_vec = tf.concat(sub_vecs, 0)

        with tf.GradientTape() as g:
            g.watch(input_vec)
            if bs == 1:
                vec2g = [self.base_graph(input_vec[n_nodes * i:n_nodes * (i + 1)], ks, ms, n_nodes) for i in range(bs)]
            else:
                vec2g = [self.base_graph(input_vec[n_nodes * i:n_nodes * (i + 1)], ks[i], ms[i], n_nodes) for i in
                         range(bs)]

            vec2g = utils_tf.data_dicts_to_graphs_tuple(vec2g)
            vec2g = utils_tf.set_zero_global_features(vec2g, 1)
            vec2g = utils_tf.set_zero_edge_features(vec2g, 1)
            output_graphs = self.graph_network(vec2g)
            global_vals = self.out_to_global(output_graphs.globals)

        dUdq = g.gradient(global_vals, input_vec)

        return dUdq

    def train_step(self, input_batch, true_batch, ks, mass):

        train_feed = {self.base_graph_tr: input_batch,
                      self.true_dq_ph: true_batch,
                      self.ks_ph: ks,
                      self.ms_ph: mass}
        train_ops = [self.loss_op_tr, self.step_op]
        loss, _ = self.sess.run(train_ops, feed_dict=train_feed)

        return loss

    def valid_step(self, input_batch, true_batch, ks, mass):

        train_feed = {self.base_graph_tr: input_batch,
                      self.true_dq_ph: true_batch,
                      self.ks_ph: ks,
                      self.ms_ph: mass}
        train_ops = self.loss_op_tr
        loss = self.sess.run(train_ops, feed_dict=train_feed)

        return loss

    def test_step(self, input_batch, true_batch, ks, mass, steps):
        # figures relegated to jupyter notebook infengine
        stored_states = [input_batch.astype(np.float32)]
        for i in range(steps):
            test_feed = {self.test_graph_ph: stored_states[-1],
                         self.test_ks_ph: ks,
                         self.test_ms_ph: mass}
            test_ops = [self.test_next_step]

            yhat = self.sess.run(test_ops, feed_dict=test_feed)
            stored_states.append(yhat[0])

        preds = tf.concat(stored_states, 0).eval(session=self.sess)

        error = mean_squared_error(preds[self.num_nodes:, :], true_batch[:, :])

        if self.output_plots is True:
            data_dir = 'data/plots/' + self.expt_name + '/' + str(self.lr) + '/' + str(self.integ_method) + '/'

            if not os.path.exists(data_dir):
                print('non existent')
                os.makedirs(data_dir)

            plt.figure(figsize=(15, 10))
            nv = preds[:, :2]
            gt = true_batch[:, :2]
            plt.scatter(nv[:, 0], nv[:, 1], label=self.deriv_method, c='blue')
            plt.scatter(gt[:, 0], gt[:, 1], label='gt', c='black', alpha=0.5)
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(self.deriv_method + str(self.lr) + ' graphic space evolution')
            plt.savefig(data_dir + 'graphic' + self.deriv_method)

            plt.figure(figsize=(15, 10))
            nv = preds[:, :2]
            gt = true_batch[:, :2]
            plt.scatter(nv[::5, 0], nv[::5, 1], label=self.deriv_method, c='blue')
            plt.scatter(gt[::5, 0], gt[::5, 1], label='gt', c='black', alpha=0.5)
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(self.deriv_method + str(self.lr) + ' graphic space evolution')
            plt.savefig(data_dir + 'graphic' + self.deriv_method + 'onetraj')

        return error, preds[self.num_nodes:]
