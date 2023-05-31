import numpy as np
# from cell_wireless import generate_channels_cell_wireless
import matplotlib.pyplot as plt


def generate_channels_cell_wireless(num_bs, num_users, num_samples, var_noise=1.0, radius=1):
    # Network: Consisting multiple pairs of Tx and Rx devices, each pair is considered an user.
    # Input:
    #     num_users: Number of users in the network
    #     num_samples: Number of samples using for the model
    #     var_noise: variance of the AWGN
    #     p_min: minimum power for each user
    # Output:
    #     Hs: channel matrices of all users in the network - size num_samples x num_users x num_users
    #        H(i,j) is the channel from Tx of the i-th pair to Rx or the j-th pair
    #     pos: position of all users in the network (?)
    #     pos[:num_bs] is the position of the BS(s)
    #     pos[num_bs:num_bs+num_users] is the position of the user(s)
    #     adj: adjacency matrix of all users in the network - only "1" if interference occurs

    print("Generating Data for training and testing")

    if num_bs != 1:
        raise Exception("Can not generate data for training and testing with more than 1 base station")
    # generate position
    dist_mat = []
    position = []

    # Calculate channel
    CH = 1 / np.sqrt(2) * (np.random.randn(num_samples, 1, num_users)
                           + 1j * np.random.randn(num_samples, 1, num_users))

    if radius == 0:
        Hs = abs(CH)
    else:
        for each_sample in range(num_samples):
            pos = []
            pos_BS = []

            for i in range(num_bs):
                r = 0.2 * radius * (np.random.rand())
                theta = np.random.rand() * 2 * np.pi
                pos_BS.append([r * np.sin(theta), r * np.cos(theta)])
                pos.append([r * np.sin(theta), r * np.cos(theta)])
            pos_user = []

            for i in range(num_users):
                r = 0.5 * radius + 0.5 * radius * np.random.rand()
                theta = np.random.rand() * 2 * np.pi
                pos_user.append([r * np.sin(theta), r * np.cos(theta)])
                pos.append([r * np.sin(theta), r * np.cos(theta)])

            pos = np.array(pos)
            pos_BS = np.array(pos_BS)
            dist_matrix = distance_matrix(pos_BS, pos_user)
            # dist_matrixp = distance_matrix(pos[1:], pos[1:])
            dist_mat.append(dist_matrix)
            position.append(pos)

        dist_mat = np.array(dist_mat)
        position = np.array(position)

        # Calculate Free space pathloss
        f = 6e9
        c = 3e8
        FSPL = 1 / ((4 * np.pi * f * dist_mat / c) ** 2)
        Hs = abs(CH * FSPL)

    adj = adj_matrix(num_users)

    return Hs, position, adj


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
