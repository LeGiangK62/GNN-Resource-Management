import cvxpy as cp
import numpy as np
from cell_wireless import generate_channels_cell_wireless


# region Verified
def cal_all_rate(num_users, power_matrix, channel_matrix, noise_var):
    # Verified
    sinr = cal_all_sinr(num_users, power_matrix, channel_matrix, noise_var)
    sumrate = np.log2(1 + sinr)
    return sumrate


def cal_all_sinr(num_users, power_matrix, channel_matrix, noise_var):
    # Verified
    all_rx_signal = np.transpose(channel_matrix) @ power_matrix
    desired_sig = np.diag(all_rx_signal)
    noise = np.ones(num_users) * noise_var
    interference = all_rx_signal - np.diag(np.diag(all_rx_signal))
    interference = np.sum(interference, 1)
    sinr = desired_sig / (interference + noise)
    return sinr


def log_approximation(num_users, power_matrix, channel_matrix, noise_var):
    # Verified
    sinr = cal_all_sinr(num_users, power_matrix, channel_matrix, noise_var)
    alpha = sinr / (1 + sinr)
    # beta = np.log2(1 + sinr) - alpha * np.log2(1 + sinr)
    beta = np.log2(1 + sinr) - np.multiply(alpha, np.log2(sinr))
    return [alpha, beta]
# endregion


def cal_cvx_rate(alpha, beta, num_users, pbar_cvx, channel_matrix, noise_var):

    # calculate sinr
    p_cvx = cp.exp(pbar_cvx)
    all_rx_signal = cp.transpose(channel_matrix) @ p_cvx
    desired_sig = cp.diag(all_rx_signal)
    noise = np.ones(num_users) * noise_var
    interference = all_rx_signal - cp.diag(cp.diag(all_rx_signal))
    interference = cp.sum(interference, 1)
    sinr = desired_sig / (interference + noise)
    # print(f'sinr: {sinr}')
    # print(f'alpha: {alpha}')
    # print(f'beta: {beta}')
    sumrate = cp.multiply(alpha, cp.log(sinr)/cp.log(2)) + beta
    # sumrate = alpha @ cp.log(sinr) + beta
    return cp.sum(sumrate)


def cal_apprx_rate(alpha, beta, num_users, pbar_cvx, channel_matrix, noise_var):

    # calculate sinr
    p_cvx = np.exp(pbar_cvx)
    sinr = cal_all_sinr(num_users, p_cvx, channel_matrix, noise_var)
    # print(f'sinr: {sinr}')
    # print(f'alpha: {alpha}')
    # print(f'beta: {beta}')
    sumrate = np.multiply(alpha, np.log(sinr)/np.log(2)) + beta
    # sumrate = alpha @ cp.log(sinr) + beta
    return np.sum(sumrate)


def log_algorithm(number_users, channel_matrix, noise_var, power_max, power_init=None):
    if power_init is None:
        power = np.ones((1, number_users)) * 0.01
    else:
        power = power_init  # initial power

    sumrate_save = []
    count = 0
    while 1:
        count = count + 1
        # Calculater/Update alpha and beta
        [Alpha, Beta] = log_approximation(number_users, power, channel_matrix, noise_var)
        # sinr = cal_all_sinr(number_users, power, channel_matrix, noise_var)
        # Solve the problem using cvx
        # convex start here
        pbar_cvx = cp.Variable(shape=(1, number_users))

        sumrate_cvx = cal_cvx_rate(Alpha, Beta, number_users, power, channel_matrix, noise_var)
        objective = cp.Maximize(sumrate_cvx)
        # constraint = [
        #     1 <= cp.exp(pbar_cvx)
        # ]  # Add contraints here, p >= 0 ?
        constraint = [
            # 1 <= pbar_cvx,
            cp.exp(pbar_cvx) <= power_max,
            # cp.sum(pbar_cvx) <= number_users
        ]  # Add contraints here, p >= 0 ?

        cvx_prob = cp.Problem(objective, constraint)
        result = cvx_prob.solve()
        # convex end here
        Pbar_val = pbar_cvx.value
        power = np.exp(Pbar_val)

        # calculate the sum rate - real sumrate
        # sum_approx = cal_apprx_rate(Alpha, Beta, number_users, Pbar_val, channel_matrix, noise_var)
        rate_all = cal_all_rate(number_users, power, channel_matrix, noise_var)
        sum_rate = np.sum(rate_all)
        sumrate_save.append(sum_rate)

        # check the convergence condition
        # if len(sumrate_save) > 2:
        #     # print(np.abs(sumrate_save[-1] - sumrate_save[-2])) ## why = 0 ?
        #     if np.abs(sumrate_save[-1] - sumrate_save[-2]) / sumrate_save[-2] <= 0.0001:
        #         break
        if count == 1000:
            break

    return power, sumrate_save, result, pbar_cvx


if __name__ == '__main__':
    K = 1  # number of BS(s)
    N = 10  # number of users
    R = 10  # radius
    p_mtx = np.ones((1, N)) * 0.01
    p_max = np.ones((1, N)) * 2

    num_train = 2  # number of training samples
    num_test = 10  # number of test samples

    reg = 1e-2
    pmax = 1
    var_db = 10
    var = 1 / 10 ** (var_db / 10)
    X_train, pos_train, adj_train = generate_channels_cell_wireless(K, N, num_train, var, R)

    # sinr_test = cal_all_sinr(N, p_mtx, X_train[0,:], var)
    # print(sinr_test)
    power_sol, all_sum, solution, pbar = log_algorithm(N, X_train[0, :], var, p_max, p_mtx)
    # #
    print(power_sol)
    # print(len(all_sum))
    # print(pbar.value)





