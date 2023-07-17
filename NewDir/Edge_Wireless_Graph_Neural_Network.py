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
        x1 = torch.ones(num_users, 1)
        x2 = torch.ones(num_users, 1)  # power allocation
        x3 = torch.ones((num_users, 1), dtype=torch.int32)  # ap selection?
        user_feat = torch.cat((x1,x2,x3),1)  # features of user_node
        ap_feat = torch.zeros(num_aps, num_aps_features)  # features of user_node
        edge_feat_uplink = channel_matrices[i, :, :].reshape(-1, 1)
        edge_feat_downlink = channel_matrices[i, :, :].reshape(-1, 1)
        graph = HeteroData({
            'ue': {'x': user_feat},
            'ap': {'x': ap_feat}
        })
        # Create edge types and building the graph connectivity:
        graph['ue', 'uplink', 'ap'].edge_attr = torch.tensor(edge_feat_uplink, dtype=torch.float)
        graph['ap', 'downlink', 'ue'].edge_attr = torch.tensor(edge_feat_downlink, dtype=torch.float)
        graph['ue', 'uplink', 'ap'].edge_index = torch.tensor(adj_matrix(num_users, num_aps).transpose(), dtype=torch.int64)
        graph['ap', 'downlink', 'ue'].edge_index = torch.tensor(adj_matrix(num_aps, num_users).transpose(),
                                                                dtype=torch.int64)

        # graph['ap', 'downlink', 'ue'].edge_attr  = torch.tensor(edge_feat_downlink, dtype=torch.float)
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
#             'ue': {'x': user_feat},
#             'ap': {'x': ap_feat}
#         })
#
#         # Create edge types and building the graph connectivity:
#         graph['ue', 'up', 'ap'].edge_index = torch.tensor(edge_feat, dtype=torch.float)
#         # graph['ap', 'down', 'ue'].edge_index = torch.tensor(edge_feat, dtype=torch.float)
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

        self.lin1 = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        original = x_dict['ue'].clone
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        original = x_dict['ue']  # not original
        power = self.lin(x_dict['ue'])
        ap_selection = self.lin1(x_dict['ue'])
        ap_selection = torch.abs(ap_selection).int()

        out = torch.cat((original[:, 1].unsqueeze(-1), power[:, 1].unsqueeze(-1), ap_selection[:, 1].unsqueeze(-1)), 1)
        return out


#endregion


#region Training and Testing functions
def loss_function(output, batch, is_train=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_user = batch['ue']['x'].shape[0]
    num_ap = batch['ap']['x'].shape[0]
    ##
    channel_matrix = batch['ue', 'ap']['edge_attr']
    ##
    power_max = output[:, 0]
    power = output[:, 1]
    ap_selection = output[:, 2]
    # power_max = batch['ue']['x'][:, 0]
    # power = batch['ue']['x'][:, 1]
    # ap_selection = batch['ue']['x'][:, 2]
    ##
    ap_selection = ap_selection.int()
    index = torch.arange(num_user)

    G = torch.reshape(channel_matrix, (-1, num_ap, num_user))
    # P = torch.reshape(power, (-1, num_ap, num_user)) #* p_max
    P = torch.zeros_like(G, requires_grad=True).clone()
    P[0, ap_selection[index], index] = power_max * power
    ##
    # new_noise = torch.from_numpy(noise_matrix).to(device)
    desired_signal = torch.sum(torch.mul(P, G), dim=1).unsqueeze(-1)
    G_UE = torch.sum(G, dim=2).unsqueeze(-1)
    all_signal = torch.matmul(P.permute((0,2,1)), G_UE)
    interference = all_signal - desired_signal #+ new_noise
    rate = torch.log(1 + torch.div(desired_signal, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))
    mean_power = torch.mean(torch.sum(P.permute((0,2,1)), 1))

    if is_train:
        return torch.neg(sum_rate / mean_power)
    else:
        return sum_rate / mean_power



def train(data_loader):
    model.train()
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_examples = total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        batch = batch.to(device_type)
        # Add counting part here => to reshape output when calculate loss
        # K = d_train.shape[-1]
        # n = len(g.nodes['UE'].data['feat'])
        # bs = len(g.nodes['UE'].data['feat']) // K
        # batch_size = batch['ue'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        tmp_loss = loss_function(out, data, True)
        tmp_loss.backward()
        optimizer.step()
        #total_examples += batch_size
        total_loss += float(tmp_loss) #* batch_size

    return total_loss #/ total_examples


def test(data_loader):
    model.eval()
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_examples = total_loss = 0
    for batch in data_loader:
        batch = batch.to(device_type)
        # batch_size = batch['ue'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        tmp_loss = loss_function(out, batch, False)
        #total_examples += batch_size
        total_loss += float(tmp_loss) #* batch_size

    return total_loss #/ total_examples
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

    batchSize = 1

    train_loader = DataLoader(train_data, batchSize, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batchSize, shuffle=True, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = train_data[0]
    data = data.to(device)

    model = HetNetGNN(data, hidden_channels=64, out_channels=4, num_heads=2, num_layers=1)
    model = model.to(device)

    # # # print(data.edge_index_dict)
    # with torch.no_grad():
    #     output = model(data.x_dict, data.edge_index_dict)
    # print(output)
    # print(data)

    # data = test_data[0]
    # data = data.to(device)
    #
    # with torch.no_grad():
    #     output = model(data.x_dict, data.edge_index_dict)
    #     print(output)



    # Training and testing
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    # Test
    model.train()
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device_type)
        # batch_size = batch['ue'].batch_size
        break
    # print(batch)
    out = model(batch.x_dict, batch.edge_index_dict)
    print(out)
    #
    # for epoch in range(1, 101):
    #     loss = train(train_loader)
    #     test_acc = test(test_loader)
    #     print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Test Reward: {test_acc:.4f}')

