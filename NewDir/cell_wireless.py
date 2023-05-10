import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from NewDir.reImplement import GCNet


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
    for each_sample in range(num_samples):
        pos = []
        pos_BS = []

        for i in range(num_bs):
            r = radius * np.random.rand()
            theta = np.random.rand() * 2 * np.pi
            pos_BS.append([r * np.sin(theta), r * np.cos(theta)])
            pos.append([r * np.sin(theta), r * np.cos(theta)])
        pos_user = []

        for i in range(num_users):
            r = radius + radius * np.random.rand()
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

    # Calculate channel
    CH = 1 / np.sqrt(2) * (np.random.randn(num_samples, 1, num_users)
                           + 1j * np.random.randn(num_samples, 1, num_users))
    Hs = abs(CH * FSPL)

    adj = adj_matrix(num_users)

    return Hs, position, adj


# Build adjacency between node? which nodes interacts with each other.
# default = all nodes (pair) interaction with each other (interference)
def adj_matrix(num_users):
    adj = []
    for i in range(num_users):
        for j in range(num_users):
            if not (i == j):
                adj.append([i, j])
    return np.array(adj)


# GNN configuration

def loss_function(data, out, device_type,):
    power = out[:, 2]
    disired_channels = out[:,0]

    num_user = out.shape[0]
    noise_var = out[1,1]
    power = torch.reshape(power, (-1, num_user, 1))

    abs_H = data.y
    abs_H_2 = torch.pow(abs_H, 2)
    rx_power = torch.mul(abs_H_2, power)
    mask = torch.eye(num_user)
    mask = mask.to(device_type)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + noise_var
    rate = torch.log(1 + torch.div(valid_rx_power, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))

    p_constraint = torch.sum(1)

    loss = torch.neg(sum_rate) + p_constraint
    return loss


def model_training(num_user, noise_var, model, train_load, device_type, num_samples, optimizer):
    model.train()

    total_loss = 0
    for data in train_load:
        data = data.to(device_type)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(data, out, num_user, device_type, noise_var)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / num_samples


def model_testing(num_user, noise_var, model, test_load, device_type, num_test):
    model.eval()

    total_loss = 0
    for data in test_load:
        data = data.to(device_type)
        with torch.no_grad():
            out = model(data)
            loss = loss_function(data, out, num_user, device_type, noise_var)
            total_loss += loss.item() * data.num_graphs
    return total_loss / num_test


def graph_build(channel_matrix, adjacency_matrix):
    num_user = channel_matrix.shape[1]
    x = np.concatenate((
        np.transpose(channel_matrix),
        var_train,
        np.ones((num_user, 1))
    ),
        axis=1
    )
    edge_index = adjacency_matrix
    edge_attr = []
    for each_interfence in adjacency_matrix:
        tx = each_interfence[0]
        rx = each_interfence[1]
        tmp = [channel_matrix[tx, rx], channel_matrix[rx, tx]]
        edge_attr.append(tmp)
    # y =
    # pos =
    data = Data(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                # y=torch.tensor(y, dtype=torch.float),
                # pos=torch.tensor(pos, dtype=torch.float)
                )
    return data


def process_data(channel_matrices):
    num_samples = channel_matrices.shape[0]
    num_user = channel_matrices.shape[1]
    data_list = []
    adj = adj_matrix(num_user)
    for i in range(num_samples):
        data = graph_build(channel_matrix=channel_matrices[i],
                           adjacency_matrix=adj
                           )
        data_list.append(data)
    return data_list


if __name__ == '__main__':

    a = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    b = a
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    bt = torch.t(b)
    print(bt)
    print(b)
    print(torch.mul(a,b).shape)
    print(torch.mul(a,bt).shape)
    # K = 1   # number of BS(s)
    # N = 10  # number of users
    # R = 10  # radius
    #
    # num_train = 100  # number of training samples
    # num_test = 10   # number of test samples
    #
    # var_db = 10
    # var = 1 / 10 ** (var_db / 10)
    #
    # var_train = np.ones((num_train,1)) * var
    # X_train, pos_train, adj_train = generate_channels_cell_wireless(K, N, num_train, var, R)
    # X_test, pos_test, adj_test = generate_channels_cell_wireless(K, N, num_test, var, R)
    # # print(channel_matrices.shape)
    # # print(positions.shape)
    # # print(adj_matrix.shape)
    # #
    # # gcn_model = GCNet()
    #
    # train_data = process_data(X_train)
    # test_data = process_data(X_test)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.is_available())
    # gcn_model = GCNet().to(device)
    # optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    # train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1)
    # test_loader = DataLoader(test_data, batch_size=2000, shuffle=False, num_workers=1)
    #
    # for epoch in range(1, 200):
    #     loss1 = model_training(N, var, gcn_model, train_loader, device, num_train, optimizer)
    #     if epoch % 8 == 0:
    #         loss2 = model_testing(N, var, gcn_model, test_loader, device, num_train)
    #         print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
    #             epoch, loss1, loss2))
    #     scheduler.step()
    #
    # torch.save(gcn_model.state_dict(), 'model.pth')
