import numpy as np
from cell_wireless import generate_channels_cell_wireless
import matplotlib.pyplot as plt

from log_algorithm_resource import log_algorithm


def draw_network(position, radius):
    fig, ax = plt.subplots()
    circle = plt.Circle((0, 0), radius, fill=False, color='blue')
    ax.set_aspect('equal', adjustable='box')
    ax.scatter(position[0][0][0], position[0][0][1], color='green')
    ax.scatter(
        [node[0] for node in position[0][1:]],
        [node[1] for node in position[0][1:]],
        color='red'
    )
    ax.add_patch(circle)
    plt.show()


def wmmse_cell_network(channel_matrix, power_matrix, weight_matrix, p_max, noise, epsilon=1e-1):
    print("Solving the cell network problem with WMMSE")
    power = np.sqrt(power_matrix)

    all_rx_signal = channel_matrix.transpose(0, 2, 1) @ power
    desired_power = np.diagonal(all_rx_signal, axis1=1, axis2=2)
    interference = np.square(all_rx_signal)
    interference = np.sum(interference, 2)  # interfernce at each UE => sum of columns
    # Init the U and W
    U = np.divide(desired_power, interference + noise)
    W = 1 / (1 - (U * desired_power))
    # The main loop
    count = 1

    while 1:
        # Calculate the V
        V_Prev = power
        all_rx_signal = channel_matrix.transpose(0, 2, 1) @ U
        desired_power = np.diagonal(all_rx_signal, axis1=1, axis2=2)
        desired_power = weight_matrix * W * desired_power
        interference = np.square(all_rx_signal)
        interference = weight_matrix * interference * W
        interference = np.sum(interference, 2)

        V = desired_power / interference

        # setting V for constraints p_max
        V = np.minimum(V, np.sqrt(p_max)) + np.maximum(V, np.zeros(V.shape)) - V

        # Update U and W
        all_rx_signal = channel_matrix.transpose(0, 2, 1) @ V
        desired_power = np.diagonal(all_rx_signal, axis1=1, axis2=2)
        interference = np.square(all_rx_signal)
        interference = np.sum(interference, 2)
        U = np.divide(desired_power, interference + noise)
        W = 1 / (1 - (U * desired_power))

        count = count + 1

        # Check break condition
        if np.linalg.norm(V - V_Prev) < epsilon or count == 100:
            break

    # print(f'The total loop: {count}')
    return np.square(V)


def check_convergence(count):
    if count == 1000:
        return True
    else:
        return False


if __name__ == '__main__':
    K = 1  # number of BS(s)
    N = 3  # number of users
    R = 0  # radius
    p_mtx = np.ones((1, K, N)) * 10
    p_max = np.ones((1, K, N)) * 4

    num_train = 1  # number of training samples
    num_test = 10  # number of test samples

    reg = 1e-2
    pmax = 1
    var_db = 10
    var = 1 / 10 ** (var_db / 10)
    weight = np.random.rand(1, K, N)

    X_train, pos_train, adj_train = generate_channels_cell_wireless(K, N, num_train, var, R)

    # X_train = np.array([[[0.49632743, 0.45383659, 0.44659692]]])

    weights_matrix = np.array([[[1, 1, 1]]])
    power = np.array([[[1, 2, 4]]])
    var_noise = np.array([[[0.1, 0.1, 0.1]]])

    p_wmmse = wmmse_cell_network(X_train, power, weights_matrix, p_max, var_noise)
    print(p_wmmse)

    # # draw_network(pos_train, R)
    # Using log_approximation algorithm
    # p_mtx = np.ones((1, N)) * 0.01
    # p_max = np.ones((1, N)) * 2
    # p_log,  all_sum, solution, pbar = log_algorithm(N, X_train[0, :], var, p_max, p_mtx)
    # print(p_log)

