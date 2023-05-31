import numpy as np
from log_algorithm_resource import cal_all_sinr, cal_all_rate
from cell_wireless import generate_channels_cell_wireless


K = 1  # number of BS(s)
N = 3  # number of users
R = 0  # radius
p_mtx = np.ones((1, N)) * 0.01

num_train = 2  # number of training samples
num_test = 10  # number of test samples

reg = 1e-2
pmax = 1
var_db = 10
var = 1 / 10 ** (var_db / 10)


num_users = N
power_matrix = np.array([[1, 2, 4]])
noise_var = var
# X_train, pos_train, adj_train = generate_channels_cell_wireless(K, N, num_train, var, R)
#
#
# print(X_train[0, :])
X_train = np.array([[[0.90751557, 1.023845,   1.27434965]]])
# sinr_test = cal_all_sinr(N, p_mtx, X_train[0, :], var)
sinr = cal_all_sinr(N, power_matrix, X_train[0, :], var)
print(sinr)
print(1 + sinr)
# sumrate = np.log2(1 + sinr)
# print(sumrate)


alpha = sinr / (1 + sinr)
# beta = np.log2(1 + sinr) - alpha * np.log2(1 + sinr)
beta = np.log2(1 + sinr) - np.multiply(alpha, np.log2(sinr))

print(f'alpha: {alpha}, beta: {beta}')



