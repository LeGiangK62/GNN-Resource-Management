import torch
import numpy as np
import scipy  # for testing


from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Linear, HGTConv

from WSN_GNN import generate_channels_wsn


#region Create HeteroData from the wireless system
def convert_to_hetero_data(channel_matrices):
    graph_list = []
    num_sam, num_aps, num_users = channel_matrices.shape
    for i in range(num_sam):
        user_feat = torch.randn(num_users, num_users_features)  # features of user_node
        ap_feat = torch.randn(num_aps, num_aps_features)  # features of user_node
        edge_feat_uplink = channel_matrices[i, :, :].reshape(-1, 1)
        edge_feat_downlink = channel_matrices[i, :, :].reshape(-1, 1)
        graph = HeteroData({
            'user': {'x': user_feat},
            'ap': {'x': ap_feat}
        })
        # Create edge types and building the graph connectivity:
        graph['user', 'uplink', 'ap'].edge_attr = torch.tensor(edge_feat_uplink, dtype=torch.float)
        graph['user', 'uplink', 'ap'].edge_index = torch.tensor(adj_matrix(num_users, num_aps).transpose(), dtype=torch.int64)
        graph['ap', 'downlink', 'user'].edge_index = torch.tensor(adj_matrix(num_aps, num_users).transpose(),
                                                                dtype=torch.int64)

        # graph['ap', 'downlink', 'user'].edge_attr  = torch.tensor(edge_feat_downlink, dtype=torch.float)
        graph_list.append(graph)
    return graph_list


def adj_matrix(num_from, num_dest):
    adj = []
    for i in range(num_from):
        for j in range(num_dest):
            adj.append([i, j])
    return np.array(adj)


#region Class Pending
# class HeteroWirelessData(HeteroData):
#     def __init__(self, channel_matrices):
#         self.channel_matrices = channel_matrices
#         self.adj, self.adj_t = self.get_cg()
#         self.num_users = channel_matrices.shape[2]
#         self.num_aps = channel_matrices.shape[1]
#         self.num_samples = channel_matrices.shape[0]
#         self.graph_list = self.build_all_graph()
#         super().__init__(name="ResourceAllocation")
#
#     def get_cg(self):
#         # The graph is a fully connected bipartite graph
#         self.adj = []
#         self.adj_t = []
#         for i in range(0, self.num_users):
#             for j in range(0, self.num_aps):
#                 self.adj.append([i, j])
#                 self.adj_t.append([j, i])
#         return self.adj, self.adj_t
#
#     def __len__(self):
#         # 'Denotes the total number of samples'
#         return self.num_samples
#
#     def __getitem__(self, index):
#         # 'Generates one sample of data'
#         # Select sample
#         return self.graph_list[index], self.direct[index], self.cross[index]
#
#     # @staticmethod
#     def build_graph(self, index):
#         user_feat = torch.zeros(num_users, num_users_features)  # features of user_node
#         ap_feat = torch.zeros(num_aps, num_aps_features)  # features of user_node
#         edge_feat = self.channel_matrices[index, :, :]
#         graph = HeteroData({
#             'user': {'x': user_feat},
#             'ap': {'x': ap_feat}
#         })
#
#         # Create edge types and building the graph connectivity:
#         graph['user', 'up', 'ap'].edge_index = torch.tensor(edge_feat, dtype=torch.float)
#         # graph['ap', 'down', 'user'].edge_index = torch.tensor(edge_feat, dtype=torch.float)
#         return graph
#
#     def build_all_graph(self):
#         self.graph_list = []
#         n = self.num_samples  # number of samples in dataset
#         for i in range(n):
#             graph = self.build_graph(i)
#             self.graph_list.append(graph)
#         return self.graph_list
#
#endregion


#region Build Heterogeneous GNN
class HetNetGNN(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['user'])


#endregion


#region Training and Testing functions
def loss_function(data, out, num_ap, num_user, noise_matrix, p_max, train = True, isLog=False):
    # Loss function only takes data and the output to calculate energy efficiency

    G = torch.reshape(out[:, 0], (-1, num_ap, num_user))  #/ noise
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # how to get channel from data and output
    P = torch.reshape(out[:, 2], (-1, num_ap, num_user)) * p_max
    # ## ap selection part
    # ap_select = torch.reshape(out[:, 1], (-1, num_ap, num_user))
    # P = torch.mul(P, ap_select)
    # ##
    desired_signal = torch.sum(torch.mul(P,G), dim=2).unsqueeze(-1)
    P_UE = torch.sum(P, dim=1).unsqueeze(-1)
    all_received_signal = torch.matmul(G, P_UE)
    new_noise = torch.from_numpy(noise_matrix).to(device)
    interference = all_received_signal - desired_signal + new_noise
    rate = torch.log(1 + torch.div(desired_signal, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))
    mean_power = torch.mean(torch.sum(P_UE, 1))

    if(isLog):
      print(f'Channel Coefficient: {G}')
      print(f'Power: {P}')
      print(f'desired_signal: {desired_signal}')
      print(f'P_UE: {P_UE}')
      print(f'all_received_signal: {all_received_signal}')
      print(f'interference: {interference}')

    if train:
        return torch.neg(sum_rate/mean_power)
    else:
        return sum_rate/mean_power


def train(device_type, data_loader):
    model.train()

    total_examples = total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        batch = batch.to(device_type)
        batch_size = batch['user'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = loss_function(data, out)
        loss.backward()
        optimizer.step()
        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


def test(device_type, data_loader):
    model.eval()

    total_examples = total_loss = 0
    for batch in data_loader:
        batch = batch.to(device_type)
        batch_size = batch['user'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = loss_function(data, out)
        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples
#endregion

if __name__ == '__main__':
    K = 3  # number of APs
    N = 5  # number of nodes
    R = 10  # radius

    num_users_features = 3
    num_aps_features = 3

    num_train = 2  # number of training samples
    num_test = 4  # number of test samples

    reg = 1e-2
    pmax = 1
    var_db = 10
    var = 1 / 10 ** (var_db / 10)
    var_noise = 10e-11

    power_threshold = 2.0

    X_train, noise_train, pos_train, adj_train, index_train = generate_channels_wsn(K, N, num_train, var_noise, R)
    X_test, noise_test, pos_test, adj_test, index_test = generate_channels_wsn(K + 1, N + 10, num_test, var_noise, R)

    # Maybe need normalization here
    train_data = convert_to_hetero_data(X_train)
    test_data = convert_to_hetero_data(X_test)

    batchSize = 100
    train_loader = DataLoader(train_data, batchSize, shuffle=True, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = train_data[0]
    data = data.to(device)

    model = HetNetGNN(data, hidden_channels=64, out_channels=4, num_heads=2, num_layers=1)
    model = model.to(device)

    # # print(data.edge_index_dict)
    with torch.no_grad():
        output = model(data.x_dict, data.edge_index_dict)
        print(output)

    data = test_data[0]
    data = data.to(device)

    with torch.no_grad():
        output = model(data.x_dict, data.edge_index_dict)
        print(output)

    # Training and testing
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
