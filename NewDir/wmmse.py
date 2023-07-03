import numpy as np
# from cell_wireless import generate_channels_cell_wireless
import matplotlib.pyplot as plt


def sum_rate_calculation(channel_matrices, power_matrices, weight_matrices, noise_matrices):
    num_sample = channel_matrices.shape[0]
    num_node = channel_matrices.shape[1]
    all_rx_signal = channel_matrices.transpose(0, 2, 1) @ power_matrices
    desired_power = np.diagonal(all_rx_signal, axis1=1, axis2=2)
    desired_power = np.expand_dims(desired_power, axis=1)

    diagonal_matrix = np.zeros((num_sample, num_node, num_node))
    for i in range(num_sample):
        np.fill_diagonal(diagonal_matrix[i], desired_power[i])

    interference = all_rx_signal - diagonal_matrix
    interference = np.sum(interference, 2)
    interference = np.expand_dims(interference, axis=1)
    sinr = np.divide(desired_power, interference + noise_matrices)
    sumrate = np.log2(1 + sinr)
    return np.sum(sumrate,2)


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
    desired_power = np.expand_dims(desired_power, axis=1)
    interference = np.square(all_rx_signal)
    interference = np.sum(interference, 2)  # interfernce at each UE => sum of columns
    interference = np.expand_dims(interference, axis=1)
    U = np.divide(desired_power, interference + noise)
    W = 1 / (1 - (U * desired_power))
    # The main loop
    count = 1

    while 1:
        # Calculate the V
        V_Prev = power
        all_rx_signal = channel_matrix.transpose(0, 2, 1) @ U
        desired_power = np.diagonal(all_rx_signal, axis1=1, axis2=2)
        desired_power = np.expand_dims(desired_power, axis=1)
        desired_power = weight_matrix * W * desired_power
        interference = np.square(all_rx_signal)
        wei_exp = np.tile(weight_matrix, (1, 10, 1))
        W_exp = np.tile(W, (1, 10, 1))
        interference = wei_exp * interference * W_exp
        interference = np.sum(interference, 2)
        interference = np.expand_dims(interference, axis=1)

        V = desired_power / interference

        # setting V for constraints p_max
        V = np.minimum(V, np.sqrt(p_max)) + np.maximum(V, np.zeros(V.shape)) - V

        # Update U and W
        all_rx_signal = channel_matrix.transpose(0, 2, 1) @ V
        desired_power = np.diagonal(all_rx_signal, axis1=1, axis2=2)
        desired_power = np.expand_dims(desired_power, axis=1)
        interference = np.square(all_rx_signal)
        interference = np.sum(interference, 2)
        interference = np.expand_dims(interference, axis=1)
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
