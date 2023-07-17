from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing, HeteroConv
from torch_geometric.typing import EdgeType, Metadata, NodeType, SparseTensor
from torch_geometric.utils import softmax


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=True), ReLU())  # , BN(channels[i]))
        for i in range(1, len(channels))
    ])


class EdgeConv(MessagePassing):
    def __init__(self, input_dim, node_dim, aggr='mean', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr)
        self.lin_node = MLP([input_dim, 32])
        self.lin_edge = MLP([input_dim, 32])
        self.res_lin = nn.Linear(node_dim, 32)

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Union[Dict[EdgeType, Tensor],
            Dict[EdgeType, SparseTensor]]
    ) -> Dict[NodeType, Optional[Tensor]]:
        # How to get the edge attributes from only the index?

        out_dict = {}
        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            # aggregate information to the destination node
            src_type, _, dst_type = edge_type
            out = self.propagate(edge_index, #)
            out_dict[dst_type].append(out)

        # Iterate over node-types:
        for node_type, outs in out_dict.items():
            out = #self.lin_node()
            out_dict[node_type].append(out)
        return out_dict


    def message(self, x_j: Tensor) -> Tensor:
        # This function is called when we use self.propagate - Used the given parameters too.
        return x_j

    # def forward(self, x, edge_index, edge_attr):
    #     x = self.lin(edge_attr)  # Process edge attributes with MLP
    #     return self.propagate(edge_index, x=x)

    def update(self, aggr_out, x):
        # This is where I concat into node feature?
        # return aggr_out + self.res_lin(x)

        out = torch.cat((x[:, 1].unsqueeze(-1), power[:, 1].unsqueeze(-1), ap_selection[:, 1].unsqueeze(-1)), 1)
        # Output must be the x_dict?
        return out


class RGCN(nn.Module):
    def __init__(self, num_layers):
        # The only things need to fix here are the dimensions
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('ue', 'uplink', 'ap'): EdgeConv(input_dim=4, node_dim=1),
                ('ap', 'downlink', 'ue'): EdgeConv(input_dim=4, node_dim=1)
            }, aggr='sum')
            self.convs.append(conv)
        ################################
        # self.conv1 = EdgeConv(input_dim=4, node_dim=1)
        # self.conv2 = EdgeConv(input_dim=66, node_dim=32)
        # self.conv3 = EdgeConv(input_dim=66, node_dim=32)
        # self.mlp = MLP([32, 16])
        # self.mlp = Seq(*[self.mlp, Seq(Lin(16, 1), Sigmoid())])

    def forward(self, graph):
        for conv in self.convs:
            x_dict = conv(graph.x_dict, graph.edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.mlp(x_dict['ue'])
        ##################################
        # x, edge = graph.x_dict, graph.edge_index_dict
        #
        # x = self.conv1(x, edge)
        # x = self.conv2(x, edge)
        # x = self.conv3(x, edge)
        # x = self.mlp(x['ue'])
        #
        # return x

# Note: The input data format in `torch_geometric` might differ from what is used in the original code.
# Make sure to preprocess your input data accordingly to create `data` objects for the RGCN model.
