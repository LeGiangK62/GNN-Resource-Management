import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from torch_geometric.data import Data

from reImplement import GCNet
from setup_arguments import setup_args


def generate_channels_wsn(num_ap, num_user, num_samples, var_noise=1.0, radius=1):
    print("Generating Data for training and testing")

    # if num_ap != 1:
    #     raise Exception("Can not generate data for training and testing with more than 1 base station")
    # generate position
    dist_mat = []
    position = []
    index_user = np.tile(np.arange(N), (K,1))
    index_ap = np.tile(np.arange(K).reshape(-1, 1), (1, N))

    index = np.array([index_user, index_ap])

    # Calculate channel
    CH = 1 / np.sqrt(2) * (np.random.randn(num_samples, 1, num_user)
                           + 1j * np.random.randn(num_samples, 1, num_user))

    if radius == 0:
        Hs = abs(CH)
    else:
        for each_sample in range(num_samples):
            pos = []
            pos_BS = []

            for i in range(num_ap):
                r = radius * (np.random.rand())
                theta = np.random.rand() * 2 * np.pi
                pos_BS.append([r * np.sin(theta), r * np.cos(theta)])
                pos.append([r * np.sin(theta), r * np.cos(theta)])
            pos_user = []

            for i in range(num_user):
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
        # f = 2e9
        # c = 3e8
        # FSPL_old = 1 / ((4 * np.pi * f * dist_mat / c) ** 2)
        FSPL = - (120.9 + 37.6 * np.log10(dist_mat/1000))
        FSPL = 10 ** (FSPL / 10)

        # print(f'FSPL_old:{FSPL_old.sum()}')
        # print(f'FSPL_new:{FSPL.sum()}')
        Hs = abs(CH * FSPL)

    adj = adj_matrix(num_user * num_ap)

    return Hs, position, adj, index


def adj_matrix(num_nodes):
    adj = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if not (i == j):
                adj.append([i, j])
    return np.array(adj)


def draw_network(position, radius, num_user, num_ap):
    ap_pos, node_pos = np.split(position, [num_ap])

    fig, ax = plt.subplots()
    circle = plt.Circle((0, 0), radius, fill=False, color='blue')
    ax.set_aspect('equal', adjustable='box')
    ax.scatter(
        [node[0] for node in ap_pos],
        [node[1] for node in ap_pos],
        color='blue'
    )
    ax.scatter(
        [node[0] for node in node_pos],
        [node[1] for node in node_pos],
        color='red'
    )
    ax.add_patch(circle)
    plt.show()


def graph_build(channel_matrix, index_matrix):
    num_user, num_ap = channel_matrix.shape
    adjacency_matrix = adj_matrix(num_user * num_ap)

    index_user = np.reshape(index_matrix[0], (-1, 1))
    index_ap = np.reshape(index_matrix[1], (-1, 1))

    x1 = np.reshape(channel_matrix, (-1, 1))
    x2 = np.ones((N * K, 1)) # power max here, for each?
    x3 = np.zeros((N * K, 1))
    x = np.concatenate((x1, x2, x3),axis=1)

    edge_index = adjacency_matrix
    edge_attr = []

    for each_interference in adjacency_matrix:
        tx = each_interference[0]
        rx = each_interference[1]

        tmp = [channel_matrix[index_ap[rx][0]][index_user[tx][0]]]
#         tmp = [
#             [channel_matrix[index_ap[rx][0]][index_user[tx][0]]],
#             [channel_matrix[index_ap[tx][0]][index_user[rx][0]]]
#         ]
        edge_attr.append(tmp)

    # y = np.expand_dims(channel_matrix, axis=0)
    # pos = np.expand_dims(weights_matrix, axis=0)

    data = Data(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                # y=torch.tensor(y, dtype=torch.float),
                # pos=torch.tensor(pos, dtype=torch.float)
                )
    return data

def build_all_data(channel_matrices, index_mtx):
    num_sample = channel_matrices.shape[0]
    data_list = []
    for i in range(num_sample):
        data = graph_build(channel_matrices[i], index_mtx)
        data_list.append(data)

    return data_list

def training_loss(data, out):
    G =
    # how to get channel from data and output
    P = np.array([[1, 0, 0, 1],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]]
                 )
    desired_signal = np.sum(P * G, axis=1)
    P_UE = np.sum(P, axis=0)
    all_received_signal = G @ P_UE
    interference = all_received_signal - desired_signal
    rate = torch.log(1 + torch.div(desired_signal, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))
    loss = torch.neg(sum_rate)
    return loss


    power = out[:, 2]
    power = torch.reshape(power, (-1, num_user, 1))
    abs_H = data.y
    abs_H_2 = torch.pow(abs_H, 2)
    rx_power = torch.mul(abs_H_2, power)
    mask = torch.eye(num_user)
    mask = mask.to(device_type)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + noise_var
    rate = torch.log(1 + torch.div(valid_rx_power, interference))
    w_rate = torch.mul(data.pos, rate)
    sum_rate = torch.mean(torch.sum(w_rate, 1))
    loss = torch.neg(sum_rate)
    return loss
    return 1


def training_model():


    return 1


def testing_model():

    return 1





if __name__ == '__main__':
    args = setup_args()


    K = 3  # number of APs
    N = 3  # number of nodes
    R = 10  # radius

    num_train = 2  # number of training samples
    num_test = 10  # number of test samples

    reg = 1e-2
    pmax = 1
    var_db = 10
    var = 1 / 10 ** (var_db / 10)

    X_train, pos_train, adj_train, index_train = generate_channels_wsn(K, N, num_train, var, R)

    # draw_network(pos_train[0], R, N, K)

    print(pos_train.shape)
    print(X_train.shape)
    print(pos_train)
    print(X_train)
    # print(pos_train)


