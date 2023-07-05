import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN

from reImplement import GCNet
from setup_arguments import setup_args


def generate_channels_wsn(num_ap, num_user, num_samples, var_noise=1.0, radius=1):
    # print("Generating Data for training and testing")

    # if num_ap != 1:
    #     raise Exception("Can not generate data for training and testing with more than 1 base station")
    # generate position
    dist_mat = []
    position = []
    index_user = np.tile(np.arange(num_user), (num_ap, 1))
    index_ap = np.tile(np.arange(num_ap).reshape(-1, 1), (1, num_user))

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


class IGConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(IGConv, self).__init__(aggr='max', **kwargs)

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


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=True), ReLU())  # , BN(channels[i]))
        for i in range(1, len(channels))
    ])


class IGCNet(torch.nn.Module):
    def __init__(self):
        super(IGCNet, self).__init__()

        self.mlp1 = MLP([5, 16, 32])
        self.mlp2 = MLP([35, 16])
        self.mlp2 = Seq(*[self.mlp2, Seq(Lin(16, 1, bias=True), Sigmoid())])
        self.conv = IGConv(self.mlp1, self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        # x3 = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        # x4 = self.conv(x = x3, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        return out



def graph_build(channel_matrix, index_matrix):
    num_user, num_ap = channel_matrix.shape
    adjacency_matrix = adj_matrix(num_user * num_ap)

    index_user = np.reshape(index_matrix[0], (-1, 1))
    index_ap = np.reshape(index_matrix[1], (-1, 1))

    x1 = np.reshape(channel_matrix, (-1, 1))
    x2 = np.ones((num_user * num_ap, 1)) # power max here, for each?
    x3 = np.zeros((num_user * num_ap, 1))
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

def data_rate_calc(data, out, num_ap, num_user, train = True):
    G = torch.reshape(out[:, 0], (-1, num_ap, num_user))
    # how to get channel from data and output
    P = torch.reshape(out[:, 2], (-1, num_ap, num_user))
    desired_signal = torch.sum(torch.mul(P, G), dim=2).unsqueeze(-1)
    P_UE = torch.sum(P, dim=1).unsqueeze(-1)
    all_received_signal = torch.matmul(G, P_UE)
    interference = all_received_signal - desired_signal
    rate = torch.log(1 + torch.div(desired_signal, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))
    if train:
        # power_max = torch.reshape(out[:, 1], (-1, num_ap, num_user))
        # regularization = torch.mean(P - power_max)
        # return torch.neg(sum_rate - regularization)
        return torch.neg(sum_rate)
    else:
        return sum_rate


if __name__ == '__main__':
    # args = setup_args()


    K = 3  # number of APs
    N = 20  # number of nodes
    R = 10  # radius

    num_train = 12  # number of training samples
    num_test = 4  # number of test samples

    reg = 1e-2
    pmax = 1
    var_db = 10
    var = 1 / 10 ** (var_db / 10)

    # Generate Data for training and testing
    X_train, pos_train, adj_train, index_train = generate_channels_wsn(K, N, num_train, var, R)
    X_test, pos_test, adj_test, index_test = generate_channels_wsn(K + 1, N + 10, num_test, var, R)

    # Preparing Data in to graph structured for model
    train_data_list = build_all_data(X_train, index_train)
    test_data_list = build_all_data(X_test, index_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GCNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    train_loader = DataLoader(train_data_list, batch_size=10, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data_list, batch_size=10, shuffle=False, num_workers=1)


    # Training and Testing model
    for epoch in range(1, 140):
        total_loss = 0
        for each_data in train_loader:
            data = each_data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = data_rate_calc(data, out, K, N, train=True)
            loss.backward()
            total_loss += loss.item() + data.num_graphs
            optimizer.step()

        train_loss = total_loss / num_train

        if (epoch % 8 == 1):
            model.eval()
            total_loss = 0
            for each_data in test_loader:
                data = each_data.to(device)
                out = model(data)
                loss = data_rate_calc(data, out, K + 1, N + 10, train=False)
                total_loss += loss.item() + data.num_graphs

            test_loss = total_loss / num_test

            print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
                epoch, train_loss, test_loss))
        scheduler.step()



