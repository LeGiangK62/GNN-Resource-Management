import numpy as np
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from cell_wireless import generate_channels_cell_wireless, GCNet, adj_matrix
from wmmse import wmmse_cell_network, draw_network
from reImplement import graph_build


def supervised_loss(data, out, num_user, device_type, noise_var):
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


def unsupervised_loss(data, out, num_user, device_type, noise_var):
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


def model_train(num_user, noise_var, model, train_loader, device_type, num_samples, optimizer):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device_type)
        optimizer.zero_grad()
        out = model(data)
        loss = unsupervised_loss(data, out, num_user, device, noise_var)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / num_samples


def model_eval(num_user, noise_var, model, test_loader, device_type, num_test):
    model.eval()

    total_loss = 0
    for data in test_loader:
        data = data.to(device_type)
        with torch.no_grad():
            out = model(data)
            loss = unsupervised_loss(data, out, num_user, device, noise_var)
            total_loss += loss.item() * data.num_graphs
    return total_loss / num_test


# Turning data into Graph-structured
def graph_build_new(channel_matrix, weights_matrix, num_users, adjacency_matrix):
    # Convert a dataset (1 sample only) to graph-structured data
    # x1 = np.expand_dims(np.diag(channel_matrix), axis=1)
    # x2 = np.expand_dims(weights_matrix, axis=1)
    # x3 =np.ones((num_users, 1))
    # print(x1.shape, x2.shape, x3.shape)
    x = np.concatenate(
        (
            np.expand_dims(np.diag(channel_matrix), axis=1),
            np.expand_dims(weights_matrix, axis=1),
            np.ones((num_users, 1))
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
    y = np.expand_dims(channel_matrix, axis=0)
    pos = np.expand_dims(weights_matrix, axis=0)

    data = Data(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                y=torch.tensor(y, dtype=torch.float),
                pos=torch.tensor(pos, dtype=torch.float)
                )
    return data


def process_data(channel_matrices, weights_matrices):
    num_samples = channel_matrices.shape[0]
    num_user = channel_matrices.shape[1]
    data_list = []
    adj = adj_matrix(num_user)
    for i in range(num_samples):
        data = graph_build(channel_matrix=channel_matrices[i], weights_matrix=weights_matrices,
                           num_users=num_user, adjacency_matrix=adj
                           )
        data_list.append(data)
    return data_list


if __name__ == '__main__':
    K = 1  # number of BS(s)
    N = 3  # number of users
    R = 0  # radius
    num_train = 1  # number of training samples
    num_test = 5  # number of training samples

    var_db = 10
    var = 1 / 10 ** (var_db / 10)

    p_max = np.ones((1, K, N)) * 10
    var_noise = np.ones((1, K, N)) * var
    wei_matrix = np.ones((1, K, N))

    Xtrain, pos_train, adj_train = generate_channels_cell_wireless(K, N, num_train, var, R)
    Xtest, pos_test, adj_test = generate_channels_cell_wireless(K, N, num_test, var, R)

    # draw_network(pos_train, R)

    print(Xtrain)
    p_wmmse = wmmse_cell_network(Xtrain, p_max, wei_matrix, p_max, var_noise)
    print(p_wmmse)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    train_data = process_data(Xtrain, wei_matrix)
    test_data = process_data(Xtest, wei_matrix)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=2000, shuffle=False, num_workers=1)

    for epoch in range(1, 200):
        loss1 = model_train(N, var, model, train_loader, device, num_train, optimizer)
        if epoch % 8 == 0:
            loss2 = model_eval(N, var, model, test_loader, device, num_train)
            print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
                epoch, loss1, loss2))
        scheduler.step()

    # torch.save(model.state_dict(), 'model.pth')