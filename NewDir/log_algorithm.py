import cvxpy as cp
import numpy as np
from cell_wireless import generate_channels_cell_wireless


def cal_rate_approx(alpha, beta, sinr):
    # Need to be changed
    # sumrate = sumrate + Alpha(i) * (Pbar_cvx(i) * log(2) + log(ChannelAll(i, termAP) ^ 2) - log(sum(Inte))) + Beta(i);
    sumrate = alpha * np.log2(sinr) + beta
    return np.sum(sumrate)


def cal_all_rate(num_users, power_matrix, channel_matrix, noise_var):
    sinr = cal_all_sinr(num_users, power_matrix, channel_matrix, noise_var)
    sumrate = np.log2(1 + sinr)
    return sumrate


def cal_all_sinr(num_users, power_matrix, channel_matrix, noise_var):
    all_rx_signal = np.transpose(channel_matrix) * power_matrix.astype(float)
    desired_sig = np.diag(all_rx_signal)
    noise = np.ones((1, num_users)) * noise_var
    interference = all_rx_signal.copy()
    np.fill_diagonal(interference, 0)
    interference = np.sum(interference, 0)
    sinr = desired_sig / (interference + noise)
    return sinr


def log_approximation(num_users, power_matrix, channel_matrix, noise_var):
    # alpha = np.zeros(num_users, 1)
    # beta = np.zeros(num_users, 1)
    sinr = cal_all_sinr(num_users, power_matrix, channel_matrix, noise_var)
    alpha = sinr / (1 + sinr)
    beta = np.log2(1 + sinr) - alpha * np.log2(1 + sinr)
    return [alpha, beta]


def log_algorithm(number_users, power_init, channel_matrix, noise_var):
    power = power_init  # initial power

    sumrate_save = []
    while 1:
        # Calculater/Update alpha and beta
        [Alpha, Beta] = log_approximation(number_users, power, channel_matrix, noise_var)
        cur_sinr = cal_all_sinr(number_users, power, channel_matrix, noise_var)
        # Solve the problem using cvx
        # convex start here
        pbar_cvx = cp.Variable(shape=(1, number_users))
        sumrate_cvx = cal_rate_approx(Alpha, Beta, cur_sinr)
        objective = cp.Maximize(sumrate_cvx)
        constraint = [pbar_cvx >= 1]  # Add contraints here, p >= 0 ?
        cvx_prob = cp.Problem(objective, constraint)
        result = cvx_prob.solve()

        Pbar_cvx = pbar_cvx.value
        # convex end here

        power = np.exp(Pbar_cvx)

        # calculate the sum rate - real sumrate
        rate_all = cal_all_rate(number_users, power, channel_matrix, noise_var)
        sum_rate = np.sum(rate_all)
        sumrate_save.append(sum_rate)

        # check the convergence condition
        if len(sumrate_save) > 20:
            print(np.abs(sumrate_save[-1] - sumrate_save[-2])) ## why = 0 ?
            if np.abs(sumrate_save[-1] - sumrate_save[-2]) / sumrate_save[-2] <= 0.0001:
                break

    return power, sumrate_save


if __name__ == '__main__':
    K = 1  # number of BS(s)
    N = 10  # number of users
    R = 0  # radius
    p_mtx = np.ones((1, N)) * 0.1

    num_train = 2  # number of training samples
    num_test = 10  # number of test samples

    reg = 1e-2
    pmax = 1
    var_db = 10
    var = 1 / 10 ** (var_db / 10)
    X_train, pos_train, adj_train = generate_channels_cell_wireless(K, N, num_train, var, R)

    power, all_sum = log_algorithm(N, p_mtx, X_train[0, :], var)

    print(power)
    print(len(all_sum))





