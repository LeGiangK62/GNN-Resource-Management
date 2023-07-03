import numpy as np
from torch_geometric.nn.conv import MessagePassing
import torch
import time
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from resource_allocation import wmmse, np_sum_rate


# Create data for training and testing
def generate_channels(num_users, num_samples, var_noise=1.0, radius=1):
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
    #     adj: adjacency matrix of all users in the network - only "1" if interference occurs

    print("Generating Data for training and testing")
    # seed = time.time_ns() % (2 ** 32)
    #
    # np.random.seed(seed)

    # generate position
    pos = []
    for i in range(num_users):
        r = radius * np.random.rand()
        theta = np.random.rand() * 2 * np.pi
        pos.append([r * np.sin(theta), r * np.cos(theta)])
    pos = np.array(pos)
    dist_matrix = distance_matrix(pos, pos)
    dist_matrix[dist_matrix == 0] = 1e-10  # to skip error, fix later
    f = 6e9
    c = 3e8
    FSPL = 1 / ((4 * np.pi * f * dist_matrix / c) ** 2)

    np.fill_diagonal(FSPL, 1)

    # Calculate channel
    CH = 1 / np.sqrt(2) * (np.random.randn(num_samples, num_users, num_users)
                           + 1j * np.random.randn(num_samples, num_users, num_users))
    Hs = abs(CH * FSPL)
    adj = adj_matrix(num_users)

    return Hs, pos, adj


def draw_network(position, radius):
    fig, ax = plt.subplots()
    circle = plt.Circle((0, 0), radius, fill=False, color='blue')
    ax.set_aspect('equal', adjustable='box')
    ax.scatter([node[0] for node in position], [node[1] for node in position], color='red')
    ax.add_patch(circle)
    plt.show()


def adj_matrix(num_users):
    adj = []
    for i in range(num_users):
        for j in range(num_users):
            if not (i == j):
                adj.append([i, j])
    return adj


# Create Graph Neural Network
class GConvLayer(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(GConvLayer, self).__init__(aggr='max', **kwargs)

        self.mlp1 = mlp1
        self.mlp2 = mlp2
        # self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp)
        return torch.cat([x[:, :2], comb], dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1, self.mlp2)


def mlp(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=True), ReLU())  # , BN(channels[i]))
        for i in range(1, len(channels))
    ])


class GCNet(torch.nn.Module):

    def __init__(self, ):
        super(GCNet, self).__init__()

        self.mlp1 = mlp([4, 16, 32])
        self.mlp2 = mlp([35, 16])
        self.mlp2 = Seq(*[self.mlp2, Seq(Lin(16, 1, bias=True), Sigmoid())])
        self.conv = GConvLayer(self.mlp1, self.mlp2)

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # x3 = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        # x4 = self.conv(x = x3, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return out


def sr_loss(data, out, num_user, device_type, noise_var):
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


def training(num_user, noise_var, model, train_loader, device_type, num_samples, optimizer):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device_type)
        optimizer.zero_grad()
        out = model(data)
        loss = sr_loss(data, out, num_user, device, noise_var)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / num_samples


def testing(num_user, noise_var, model, test_loader, device_type, num_test):
    model.eval()

    total_loss = 0
    for data in test_loader:
        data = data.to(device_type)
        with torch.no_grad():
            out = model(data)
            loss = sr_loss(data, out, num_user, device, noise_var)
            total_loss += loss.item() * data.num_graphs
    return total_loss / num_test


# Turning data into Graph-structured
def graph_build(channel_matrix, weights_matrix, num_users, adjacency_matrix):
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


# Training and testing


# main

if __name__ == '__main__':

    num_u = 10  # number of users
    R = 10  # radius
    num_train = 1000  # number of training samples
    num_test = 500  # number of testing  samples
    training_epochs = 50  # number of training epochs
    trainseed = 0  # set random seed for training set
    testseed = 7  # set random seed for test set
    # print('Gaussian IC Case: K=%d, Total Samples: %d, Total Iterations: %d\n' % (K, num_H, training_epochs))
    var_db = 10
    var = 1 / 10 ** (var_db / 10)
    weights_m = np.ones(num_u)
    X_train, position_train, adj_train = generate_channels(num_users=num_u, num_samples=num_train,
                                                           var_noise=var, radius=R)

    X_test, position_test, adj_test = generate_channels(num_users=num_u, num_samples=num_test,
                                                        var_noise=var, radius=R)
    train_data = process_data(X_train, weights_m)
    test_data = process_data(X_test, weights_m)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    model = GCNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=2000, shuffle=False, num_workers=1)
    # training
    for epoch in range(1, 200):
        loss1 = training(num_u, var, model, train_loader, device, num_train, optimizer)
        if epoch % 8 == 0:
            loss2 = testing(num_u, var, model, test_loader, device, num_train)
            print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
                epoch, loss1, loss2))
        scheduler.step()

    torch.save(model.state_dict(), 'model.pth')
    ##

    # region Testing
    # Pmax = 1
    # p = wmmse(weights_m, X_test, Pmax, var)
    # print('wmmse:', np_sum_rate(X_test.transpose(0, 2, 1), p, weights_m, var))
    #
    # # create an instance of your model
    # model = GCNet().to(device)
    #
    # # load the state dictionary from the saved file
    # model.load_state_dict(torch.load('model.pth'))
    # loss2 = testing(num_u, var, model, test_loader, device, num_test)
    #
    # print('GCNet:', -loss2)
    # endregion
