import numpy as np
import torch


def batch_WMMSE2(p_int, alpha, H, Pmax, var_noise):
    print("Solving WMMSE?")
    N = p_int.shape[0]
    K = p_int.shape[1]
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros((N, K, 1))
    w = np.zeros((N, K, 1))

    mask = np.eye(K)
    rx_power = np.multiply(H, b)
    rx_power_s = np.square(rx_power)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)

    interference = np.sum(rx_power_s, 2) + var_noise
    f = np.divide(valid_rx_power, interference)
    w = 1 / (1 - np.multiply(f, valid_rx_power))
    # vnew = np.sum(np.log2(w),1)

    for ii in range(100):
        fp = np.expand_dims(f, 1)
        rx_power = np.multiply(H.transpose(0, 2, 1), fp)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
        bup = np.multiply(alpha, np.multiply(w, valid_rx_power))
        rx_power_s = np.square(rx_power)
        wp = np.expand_dims(w, 1)
        alphap = np.expand_dims(alpha, 1)
        bdown = np.sum(np.multiply(alphap, np.multiply(rx_power_s, wp)), 2)
        btmp = bup / bdown
        b = np.minimum(btmp, np.ones((N, K)) * np.sqrt(Pmax)) + np.maximum(btmp, np.zeros((N, K))) - btmp

        bp = np.expand_dims(b, 1)
        rx_power = np.multiply(H, bp)
        rx_power_s = np.square(rx_power)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
        interference = np.sum(rx_power_s, 2) + var_noise
        f = np.divide(valid_rx_power, interference)
        w = 1 / (1 - np.multiply(f, valid_rx_power))
    p_opt = np.square(b)
    return p_opt


def wmmse(weights_matrix, channels_matrix, Pmax, var_noise):
    print("Solving power allocation using WMMSE")
    N = channels_matrix.shape[0]
    K = channels_matrix.shape[1]
    p_int = Pmax * np.ones((N, K, 1))
    vnew = 0
    b = np.sqrt(p_int)
    sinr = np.zeros((N, K, 1))
    w = np.zeros((N, K, 1))

    mask = np.eye(K)
    rx_power = np.multiply(channels_matrix, b)
    rx_power_s = np.square(rx_power)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)

    interference = np.sum(rx_power_s, 2) + var_noise
    sinr = np.divide(valid_rx_power, interference)
    w = 1 / (1 - np.multiply(sinr, valid_rx_power))
    # vnew = np.sum(np.log2(w),1)
    for ii in range(100):
        fp = np.expand_dims(sinr, 1)
        rx_power = np.multiply(channels_matrix.transpose(0, 2, 1), fp)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
        bup = np.multiply(weights_matrix, np.multiply(w, valid_rx_power))
        rx_power_s = np.square(rx_power)
        wp = np.expand_dims(w, 1)
        weights_matrix_p = np.expand_dims(weights_matrix, 1)
        bdown = np.sum(np.multiply(weights_matrix_p, np.multiply(rx_power_s, wp)), 2)
        btmp = bup / bdown
        b = np.minimum(btmp, np.ones((N, K)) * np.sqrt(Pmax)) + np.maximum(btmp, np.zeros((N, K))) - btmp

        bp = np.expand_dims(b, 1)
        rx_power = np.multiply(channels_matrix, bp)
        rx_power_s = np.square(rx_power)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
        interference = np.sum(rx_power_s, 2) + var_noise
        sinr = np.divide(valid_rx_power, interference)
        w = 1 / (1 - np.multiply(sinr, valid_rx_power))
    p_opt = np.square(b)
    return p_opt


def np_sum_rate(H, p, alpha, var_noise):
  H = np.expand_dims(H, axis=-1) # (num_H, K, K)
  K = H.shape[1]
  N = H.shape[-1] # always 1???
  p = p.reshape((-1, K, 1, N))
  rx_power = np.multiply(H, p)
  rx_power = np.sum(rx_power, axis=-1)
  rx_power = np.square(abs(rx_power))
  # rx_power(k,i,j) power received by j-th user sent from i-th user in k-th data set?
  mask = np.eye(K)
  valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=1)
  # valid_rx_power(i, j) the desired signal power received by j-th user in i-th data set
  interference = np.sum(np.multiply(rx_power, 1 - mask), axis=1) + var_noise
  rate = np.log(1 + np.divide(valid_rx_power, interference))
  w_rate = np.multiply(alpha, rate)
  sum_rate = np.mean(np.sum(w_rate, axis=1))
  return sum_rate