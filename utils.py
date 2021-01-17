import numpy as np
import tensorflow as tf


### GRAPH BASED INTEGRATORS
def rk4(dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
    k1 = dt * dx_dt_fn(x_t, ks, ms, bs, nodes)
    k2 = dt * dx_dt_fn(x_t + (1 / 2) * k1, ks, ms, bs, nodes)
    k3 = dt * dx_dt_fn(x_t + (1 / 2) * k2, ks, ms, bs, nodes)
    k4 = dt * dx_dt_fn(x_t + k3, ks, ms, bs, nodes)
    x_tp1 = x_t + (1 / 6) * (k1 + k2 * 2 + k3 * 2 + k4)
    return x_tp1


def rk3(dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
    k1 = dt * dx_dt_fn(x_t, ks, ms, bs, nodes)
    k2 = dt * dx_dt_fn(x_t + (1 / 2) * k1, ks, ms, bs, nodes)
    k3 = dt * dx_dt_fn(x_t + k2, ks, ms, bs, nodes)
    x_tp1 = x_t + (1 / 6) * (k1 + k2 * 4 + k3)
    return x_tp1


def rk1(dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
    k1 = dt * dx_dt_fn(x_t, ks, ms, bs, nodes)
    x_tp1 = x_t + k1
    return x_tp1


def rk2(dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
    k1 = dt * dx_dt_fn(x_t, ks, ms, bs, nodes)
    k2 = dt * dx_dt_fn(x_t + (1 / 2) * k1, ks, ms, bs, nodes)
    x_tp1 = x_t + k2
    return x_tp1


def vi1ng(dx_dt_fn, x_t, dt):
    subdim = int(x_t.shape[1] // 2)
    q = x_t[:subdim]
    p = x_t[subdim:]

    q1 = q + dt * dx_dt_fn(tf.concat([q, p], 1), ks, ms, bs, nodes)[:subdim]
    p1 = p + dt * dx_dt_fn(tf.concat([q1, p], 1), ks, ms, bs, nodes)[subdim:]

    return tf.concat([q1, p1], 1)


def vi2(dx_dt_fn, x_t, dt):
    subdim = int(x_t.shape[1] // 2)
    q = x_t[:, :subdim]
    p = x_t[:, subdim:]

    q1 = q + dt * 0.5 * dx_dt_fn(tf.concat([q, p], 1), ks, ms, bs, nodes)[:,:subdim]
    p1 = p + dt * 0 * dx_dt_fn(tf.concat([q1, p], 1), ks, ms, bs, nodes)[:, subdim:]
    q2 = q1 + dt * 0.5 * dx_dt_fn(tf.concat([q1, p1], 1), ks, ms, bs, nodes)[:,:subdim]
    p2 = p1 + dt * dx_dt_fn(tf.concat([q2, p1], 1), ks, ms, bs, nodes)[:, subdim:]

    return tf.concat([q2, p2], 1)


def vi3(dx_dt_fn, x_t, dt):
    subdim = int(x_t.shape[1] // 2)
    q = x_t[:, :subdim]
    p = x_t[:, subdim:]

    q1 = q + dt * (-1. / 24) * dx_dt_fn(tf.concat([q, p], 1), ks, ms, bs, nodes)[:,:subdim]
    p1 = p + dt * 1 * dx_dt_fn(tf.concat([q1, p], 1), ks, ms, bs, nodes)[:, subdim:]
    q2 = q1 + dt * (3. / 4) * dx_dt_fn(tf.concat([q1, p1], 1), ks, ms, bs, nodes)[:,:subdim]
    p2 = p1 + dt * (-2. / 3) * dx_dt_fn(tf.concat([q2, p1], 1), ks, ms, bs, nodes)[:, subdim:]
    q3 = q2 + dt * (7. / 24) * dx_dt_fn(tf.concat([q2, p2], 1), ks, ms, bs, nodes)[:,:subdim]
    p3 = p2 + dt * (2. / 3) * dx_dt_fn(tf.concat([q3, p2], 1), ks, ms, bs, nodes)[:, subdim:]

    return tf.concat([q3, p3], 1)


def vi4(dx_dt_fn, x_t, ks, ms, dt, bs, nodes):
    subdim = int(x_t.shape[1] // 2)
    q = x_t[:, :subdim]
    p = x_t[:, subdim:]

    d1 = d3 = 1. / (2 - 2 ** (1. / 3))
    d2 = -(2 ** (1. / 3)) / (2 - 2 ** (1. / 3))
    d4 = 0

    c1 = c4 = 1. / (2 * (2 - 2 ** (1. / 3)))
    c2 = c3 = (1 - 2 ** (1. / 3)) / (2 * (2 - 2 ** (1. / 3)))

    q1 = q + dt * c1 * dx_dt_fn(tf.concat([q, p], 1), ks, ms, bs, nodes)[:, :subdim]
    p1 = p + dt * d1 * dx_dt_fn(tf.concat([q1, p], 1), ks, ms, bs, nodes)[:, subdim:]
    q2 = q1 + dt * c2 * dx_dt_fn(tf.concat([q1, p1], 1), ks, ms, bs, nodes)[:, :subdim]
    p2 = p1 + dt * d2 * dx_dt_fn(tf.concat([q2, p1], 1), ks, ms, bs, nodes)[:, subdim:]
    q3 = q2 + dt * c3 * dx_dt_fn(tf.concat([q2, p2], 1), ks, ms, bs, nodes)[:, :subdim]
    p3 = p2 + dt * d3 * dx_dt_fn(tf.concat([q3, p2], 1), ks, ms, bs, nodes)[:, subdim:]
    q4 = q3 + dt * c4 * dx_dt_fn(tf.concat([q3, p3], 1), ks, ms, bs, nodes)[:, :subdim]
    p4 = p3 + dt * d4 * dx_dt_fn(tf.concat([q4, p3], 1), ks, ms, bs, nodes)[:, subdim:]
    return tf.concat([q4, p4], 1)


##### NON-GRAPH BASED INTEGRATORS
def rk4ng(dx_dt_fn, x_t, dt):
    k1 = dt * dx_dt_fn(x_t)
    k2 = dt * dx_dt_fn(x_t + (1 / 2) * k1)
    k3 = dt * dx_dt_fn(x_t + (1 / 2) * k2)
    k4 = dt * dx_dt_fn(x_t + k3)
    x_tp1 = x_t + (1 / 6) * (k1 + k2 * 2 + k3 * 2 + k4)
    return x_tp1


def rk3ng(dx_dt_fn, x_t, dt):
    k1 = dt * dx_dt_fn(x_t)
    k2 = dt * dx_dt_fn(x_t + (1 / 2) * k1)
    k3 = dt * dx_dt_fn(x_t + k2)
    x_tp1 = x_t + (1 / 6) * (k1 + k2 * 4 + k3)
    return x_tp1


def rk1ng(dx_dt_fn, x_t, dt):
    k1 = dt * dx_dt_fn(x_t)
    x_tp1 = x_t + k1
    return x_tp1


def rk2ng(dx_dt_fn, x_t, dt):
    k1 = dt * dx_dt_fn(x_t)
    k2 = dt * dx_dt_fn(x_t + (1 / 2) * k1)
    x_tp1 = x_t + k2
    return x_tp1


def vi1ng(dx_dt_fn, x_t, dt):
    subdim = int(x_t.shape[1] // 2)
    q = x_t[:subdim]
    p = x_t[subdim:]

    q1 = q + dt * dx_dt_fn(tf.concat([q, p], 1))[:subdim]
    p1 = p + dt * dx_dt_fn(tf.concat([q1, p], 1))[subdim:]

    return tf.concat([q1, p1], 1)


def vi2ng(dx_dt_fn, x_t, dt):
    subdim = int(x_t.shape[1] // 2)
    q = x_t[:, :subdim]
    p = x_t[:, subdim:]

    q1 = q + dt * 0.5 * dx_dt_fn(tf.concat([q, p], 1))[:, :subdim]
    p1 = p + dt * 0 * dx_dt_fn(tf.concat([q1, p], 1))[:, subdim:]
    q2 = q1 + dt * 0.5 * dx_dt_fn(tf.concat([q1, p1], 1))[:, :subdim]
    p2 = p1 + dt * dx_dt_fn(tf.concat([q2, p1], 1))[:, subdim:]

    return tf.concat([q2, p2], 1)


def vi3ng(dx_dt_fn, x_t, dt):
    subdim = int(x_t.shape[1] // 2)
    q = x_t[:, :subdim]
    p = x_t[:, subdim:]

    q1 = q + dt * (-1. / 24) * dx_dt_fn(tf.concat([q, p], 1))[:, :subdim]
    p1 = p + dt * 1 * dx_dt_fn(tf.concat([q1, p], 1))[:, subdim:]
    q2 = q1 + dt * (3. / 4) * dx_dt_fn(tf.concat([q1, p1], 1))[:, :subdim]
    p2 = p1 + dt * (-2. / 3) * dx_dt_fn(tf.concat([q2, p1], 1))[:, subdim:]
    q3 = q2 + dt * (7. / 24) * dx_dt_fn(tf.concat([q2, p2], 1))[:, :subdim]
    p3 = p2 + dt * (2. / 3) * dx_dt_fn(tf.concat([q3, p2], 1))[:, subdim:]

    return tf.concat([q3, p3], 1)


def vi4ng(dx_dt_fn, x_t, dt):
    subdim = int(x_t.shape[1] // 2)
    q = x_t[:, :subdim]
    p = x_t[:, subdim:]

    d1 = d3 = 1. / (2 - 2 ** (1. / 3))
    d2 = -(2 ** (1. / 3)) / (2 - 2 ** (1. / 3))
    d4 = 0

    c1 = c4 = 1. / (2 * (2 - 2 ** (1. / 3)))
    c2 = c3 = (1 - 2 ** (1. / 3)) / (2 * (2 - 2 ** (1. / 3)))

    q1 = q + dt * c1 * dx_dt_fn(tf.concat([q, p], 1))[:, :subdim]
    p1 = p + dt * d1 * dx_dt_fn(tf.concat([q1, p], 1))[:, subdim:]
    q2 = q1 + dt * c2 * dx_dt_fn(tf.concat([q1, p1], 1))[:, :subdim]
    p2 = p1 + dt * d2 * dx_dt_fn(tf.concat([q2, p1], 1))[:, subdim:]
    q3 = q2 + dt * c3 * dx_dt_fn(tf.concat([q2, p2], 1))[:, :subdim]
    p3 = p2 + dt * d3 * dx_dt_fn(tf.concat([q3, p2], 1))[:, subdim:]
    q4 = q3 + dt * c4 * dx_dt_fn(tf.concat([q3, p3], 1))[:, :subdim]
    p4 = p3 + dt * d4 * dx_dt_fn(tf.concat([q4, p3], 1))[:, subdim:]
    return tf.concat([q4, p4], 1)


def create_loss_ops(true, predicted):
    loss_ops = tf.reduce_mean((true - predicted) ** 2)
    return loss_ops


def create_loss_ops(true, predicted):
    loss_ops = tf.reduce_mean((true - predicted) ** 2)
    return loss_ops


# def base_graph(input_features, ks, ms, num_nodes, extra_flag=True):
#     # Node features for graph 0.
#     if extra_flag:
#         nodes_0 = tf.concat([input_features, tf.reshape(ms, [num_nodes, 1]), tf.reshape(ks, [num_nodes, 1])], 1)
#     else:
#         nodes_0 = input_features
#
#     senders_0 = []
#     receivers_0 = []
#     edges_0 = []
#     an = np.arange(0, num_nodes, 1)
#     for i in range(len(an)):
#         for j in range(i + 1, len(an)):
#             senders_0.append(i)
#             senders_0.append(j)
#             receivers_0.append(j)
#             receivers_0.append(i)
#
#     data_dict_0 = {
#         "nodes": nodes_0,
#         "senders": senders_0,
#         "receivers": receivers_0
#     }
#
#     return data_dict_0


def nownext(train_data, ntraj, num_nodes, T_max, dt, srate, spatial_dim=4, nograph=False):
    curr_xs = []
    next_xs = []

    curr_dxs = []
    dex = int(np.ceil((T_max / dt) / (srate / dt)))
    for i in range(ntraj):
        same_batch = train_data['x'][i * dex:(i + 1) * dex, :]
        curr_x = same_batch[:-1, :]
        next_x = same_batch[1:, :]

        curr_dx = train_data['dx'][i * dex:(i + 1) * dex, :][:-1, :]
        curr_xs.append(curr_x)
        next_xs.append(next_x)
        curr_dxs.append(curr_dx)

    curr_xs = np.vstack(curr_xs)
    next_xs = np.vstack(next_xs)
    curr_dxs = np.vstack(curr_dxs)

    if nograph:
        return curr_xs, next_xs, curr_dxs
    else:
        vdim = int(spatial_dim / 2)

        new_ls = [
            np.concatenate(
                [next_xs[i].reshape(-1, vdim)[:num_nodes], next_xs[i].reshape(-1, vdim)[num_nodes:]],
                1) for i in range(ntraj * (int(dex) - 1))
        ]
        new_ls = np.vstack(new_ls)
        true_next = new_ls  # tf.convert_to_tensor(np.float32(new_ls))

        new_in = [
            np.concatenate(
                [curr_xs[i].reshape(-1, vdim)[:num_nodes], curr_xs[i].reshape(-1, vdim)[num_nodes:]],
                1) for i in range(ntraj * (int(dex) - 1))
        ]
        new_in = np.vstack(new_in)
        true_now = new_in  # tf.convert_to_tensor(np.float32(new_ls))

        new_d = [
            np.concatenate(
                [curr_dxs[i].reshape(-1, vdim)[:num_nodes], curr_dxs[i].reshape(-1, vdim)[num_nodes:]],
                1) for i in range(ntraj * (int(dex) - 1))
        ]
        new_d = np.vstack(new_d)
        true_dxnow = new_d  # tf.convert_to_tensor(np.float32(new_ls))

        return true_now, true_next, true_dxnow
