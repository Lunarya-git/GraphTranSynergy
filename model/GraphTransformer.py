"""
# -*- coding: utf-8 -*-
# @Author : Sun JJ
# @File : GraphTransformer.py
# @Time : 2024/7/24 11:16
# code is far away from bugs with the god animal protecting
#         ┌─┐       ┌─┐
#      ┌──┘ ┴───────┘ ┴──┐
#      │                 │
#      │       ───       │
#      │  ─┬┘       └┬─  │
#      │                 │
#      │       ─┴─       │
#      │                 │
#      └───┐         ┌───┘
#          │         │
#          │         │
#          │         │
#          │         └──────────────┐
#          │                        │
#          │                        ├─┐
#          │                        ┌─┘
#          │                        │
#          └─┐  ┐  ┌───────┬──┐  ┌──┘
#            │ ─┤ ─┤       │ ─┤ ─┤
#            └──┴──┘       └──┴──┘
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_max_pool as gmp

def edge_index_to_adj(edge_index):
    num_nodes = edge_index.max().item() + 1  # 推断节点数量
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0  # 如果图是无向图，需要同时设置 (dst, src)
    return adj

# Graph Transformer based model
# class GraphTransformerNet(torch.nn.Module):
#     def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=206, num_features_xt=954, output_dim=128, dropout=0.2):
#
#         super(GraphTransformerNet, self).__init__()
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         # SMILES1 graph branch
#         self.n_output = n_output
#         self.drug1_conv1 = TransformerConv(num_features_xd, num_features_xd)
#         self.drug1_conv2 = TransformerConv(num_features_xd, num_features_xd * 2)
#         self.drug1_conv3 = TransformerConv(num_features_xd * 2, num_features_xd * 4)
#         self.drug1_fc_g1 = torch.nn.Linear(num_features_xd * 4, num_features_xd * 2)
#         self.drug1_fc_g2 = torch.nn.Linear(num_features_xd * 2, output_dim)
#
#         # SMILES2 graph branch
#         self.drug2_conv1 = TransformerConv(num_features_xd, num_features_xd)
#         self.drug2_conv2 = TransformerConv(num_features_xd, num_features_xd * 2)
#         self.drug2_conv3 = TransformerConv(num_features_xd * 2, num_features_xd * 4)
#         self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 4, num_features_xd * 2)
#         self.drug2_fc_g2 = torch.nn.Linear(num_features_xd * 2, output_dim)
#
#         # DL cell features
#         self.reduction = nn.Sequential(
#             nn.Linear(num_features_xt, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )
#
#         # combined layers
#         self.fc1 = nn.Linear(3 * output_dim, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.out = nn.Linear(128, self.n_output)
#
#     def forward(self, data1, data2):
#         x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
#         x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
#
#         # adj1 = edge_index_to_adj(edge_index1).long()
#         # adj2 = edge_index_to_adj(edge_index2).long()
#
#         # deal drug1
#         x1 = self.drug1_conv1(x1, edge_index1)
#         x1 = self.relu(x1)
#         x1 = self.drug1_conv2(x1, edge_index1)
#         x1 = self.relu(x1)
#         x1 = self.drug1_conv3(x1, edge_index1)
#         x1 = self.relu(x1)
#         x1 = gmp(x1, batch1)  # global max pooling
#         x1 = self.relu(self.drug1_fc_g1(x1))
#         x1 = self.dropout(x1)
#         x1 = self.drug1_fc_g2(x1)
#         x1 = self.dropout(x1)
#
#         # deal drug2
#         x2 = self.drug2_conv1(x2, edge_index2)
#         x2 = self.relu(x2)
#         x2 = self.drug2_conv2(x2, edge_index2)
#         x2 = self.relu(x2)
#         x2 = self.drug2_conv3(x2, edge_index2)
#         x2 = self.relu(x2)
#         x2 = gmp(x2, batch2)  # global max pooling
#         x2 = self.relu(self.drug2_fc_g1(x2))
#         x2 = self.dropout(x2)
#         x2 = self.drug2_fc_g2(x2)
#         x2 = self.dropout(x2)
#
#         # deal cell
#         cell_vector = F.normalize(cell, 2, 1)
#         cell_vector = self.reduction(cell_vector)
#
#         # concat
#         xc = torch.cat((x1, x2, cell_vector), 1)
#         xc = self.fc1(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         xc = self.fc2(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         out = self.out(xc)
#         return out


# class GraphTransformerNet(torch.nn.Module):
#     def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=206, num_features_xt=954,
#                  output_dim=128, dropout=0.2, heads=6):
#         super(GraphTransformerNet, self).__init__()
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#
#         # SMILES1 graph branch
#         self.n_output = n_output
#         self.drug1_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug1_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug1_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug1_fc_g1 = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 2)
#         self.drug1_fc_g2 = torch.nn.Linear(num_features_xd * 2, output_dim)
#
#         # SMILES2 graph branch
#         self.drug2_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug2_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug2_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 2)
#         self.drug2_fc_g2 = torch.nn.Linear(num_features_xd * 2, output_dim)
#
#         # DL cell features
#         self.reduction = nn.Sequential(
#             nn.Linear(num_features_xt, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )
#
#         # combined layers
#         self.fc1 = nn.Linear(3 * output_dim, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.out = nn.Linear(128, self.n_output)
#
#     def forward(self, data1, data2):
#         x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
#         x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
#
#         # deal drug1
#         x1 = self.drug1_conv1(x1, edge_index1)
#         x1 = self.relu(x1)
#         x1 = self.drug1_conv2(x1, edge_index1)
#         x1 = self.relu(x1)
#         x1 = self.drug1_conv3(x1, edge_index1)
#         x1 = self.relu(x1)
#         x1 = gmp(x1, batch1)  # global max pooling
#         x1 = self.relu(self.drug1_fc_g1(x1))
#         x1 = self.dropout(x1)
#         x1 = self.drug1_fc_g2(x1)
#         x1 = self.dropout(x1)
#
#         # deal drug2
#         x2 = self.drug2_conv1(x2, edge_index2)
#         x2 = self.relu(x2)
#         x2 = self.drug2_conv2(x2, edge_index2)
#         x2 = self.relu(x2)
#         x2 = self.drug2_conv3(x2, edge_index2)
#         x2 = self.relu(x2)
#         x2 = gmp(x2, batch2)  # global max pooling
#         x2 = self.relu(self.drug2_fc_g1(x2))
#         x2 = self.dropout(x2)
#         x2 = self.drug2_fc_g2(x2)
#         x2 = self.dropout(x2)
#
#         # deal cell
#         cell_vector = F.normalize(cell, 2, 1)
#         cell_vector = self.reduction(cell_vector)
#
#         # concat
#         xc = torch.cat((x1, x2, cell_vector), 1)
#         xc = self.fc1(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         xc = self.fc2(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         out = self.out(xc)
#         return out


# class GraphTransformerNet(torch.nn.Module):
#     def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=206, num_features_xt=954,
#                  output_dim=128, dropout=0.2, heads=2):
#         super(GraphTransformerNet, self).__init__()
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#
#         # SMILES1 graph branch
#         self.n_output = n_output
#         self.drug1_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug1_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug1_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug1_fc_g1 = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 2)
#         self.drug1_fc_g2 = torch.nn.Linear(num_features_xd * 2, output_dim)
#
#         # Linear layers to match dimensions for residual connections
#         self.residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * 4 * heads)
#
#         # SMILES2 graph branch
#         self.drug2_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug2_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug2_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 2)
#         self.drug2_fc_g2 = torch.nn.Linear(num_features_xd * 2, output_dim)
#
#         # Linear layers to match dimensions for residual connections
#         self.residual2_fc = torch.nn.Linear(num_features_xd, num_features_xd * 4 * heads)
#
#         # DL cell features
#         self.reduction = nn.Sequential(
#             nn.Linear(num_features_xt, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )
#
#         # combined layers
#         self.fc1 = nn.Linear(3 * output_dim, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.out = nn.Linear(128, self.n_output)
#
#     def forward(self, data1, data2):
#         x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
#         x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
#
#         # deal drug1
#         residual1 = x1
#         x1 = self.drug1_conv1(x1, edge_index1)
#         x1 = self.relu(x1)  # Apply ReLU before residual connection
#         x1 = self.drug1_conv2(x1, edge_index1)
#         x1 = self.relu(x1)
#         x1 = self.drug1_conv3(x1, edge_index1)
#         x1 = self.relu(x1)
#         residual1 = self.residual1_fc(residual1)  # Adjust dimensions of residual connection
#         x1 = self.relu(x1 + residual1)  # Residual connection
#
#         x1 = gmp(x1, batch1)  # global max pooling
#         x1 = self.relu(self.drug1_fc_g1(x1))
#         x1 = self.dropout(x1)
#         x1 = self.drug1_fc_g2(x1)
#         x1 = self.dropout(x1)
#
#         # deal drug2
#         residual2 = x2
#         x2 = self.drug2_conv1(x2, edge_index2)
#         x2 = self.relu(x2)  # Apply ReLU before residual connection
#         x2 = self.drug2_conv2(x2, edge_index2)
#         x2 = self.relu(x2)
#         x2 = self.drug2_conv3(x2, edge_index2)
#         x2 = self.relu(x2)
#         residual2 = self.residual2_fc(residual2)  # Adjust dimensions of residual connection
#         x2 = self.relu(x2 + residual2)  # Residual connection
#
#         x2 = gmp(x2, batch2)  # global max pooling
#         x2 = self.relu(self.drug2_fc_g1(x2))
#         x2 = self.dropout(x2)
#         x2 = self.drug2_fc_g2(x2)
#         x2 = self.dropout(x2)
#
#         # deal cell
#         cell_vector = F.normalize(cell, 2, 1)
#         cell_vector = self.reduction(cell_vector)
#
#         # concat
#         xc = torch.cat((x1, x2, cell_vector), 1)
#         xc = self.fc1(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         xc = self.fc2(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         out = self.out(xc)
#         return out


# class GraphTransformerNet(torch.nn.Module):
#     def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=206, num_features_xt=954,
#                  output_dim=128, dropout=0.2, heads=2):
#         super(GraphTransformerNet, self).__init__()
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#
#         # SMILES1 graph branch
#         self.n_output = n_output
#         self.drug1_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug1_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug1_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug1_fc_g1 = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 2)
#         self.drug1_fc_g2 = torch.nn.Linear(num_features_xd * 2, output_dim)
#
#         # Linear layers to match dimensions for residual connections
#         self.drug1_residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * heads)
#         self.drug1_residual2_fc = torch.nn.Linear(num_features_xd * heads, num_features_xd * 2 * heads)
#         self.drug1_residual3_fc = torch.nn.Linear(num_features_xd * 2 * heads, num_features_xd * 4 * heads)
#
#         # SMILES2 graph branch
#         self.drug2_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug2_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug2_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 2)
#         self.drug2_fc_g2 = torch.nn.Linear(num_features_xd * 2, output_dim)
#
#         # Linear layers to match dimensions for residual connections
#         self.drug2_residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * heads)
#         self.drug2_residual2_fc = torch.nn.Linear(num_features_xd * heads, num_features_xd * 2 * heads)
#         self.drug2_residual3_fc = torch.nn.Linear(num_features_xd * 2 * heads, num_features_xd * 4 * heads)
#
#         # DL cell features
#         self.reduction = nn.Sequential(
#             nn.Linear(num_features_xt, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )
#
#         # combined layers
#         self.fc1 = nn.Linear(3 * output_dim, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.out = nn.Linear(128, self.n_output)
#
#     def forward(self, data1, data2):
#         x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
#         x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
#
#         # deal drug1
#         residual1 = self.drug1_residual1_fc(x1)
#         x1 = self.drug1_conv1(x1, edge_index1)
#         x1 = self.relu(x1 + residual1)  # Residual connection
#
#         residual2 = self.drug1_residual2_fc(x1)
#         x1 = self.drug1_conv2(x1, edge_index1)
#         x1 = self.relu(x1 + residual2)  # Residual connection
#
#         residual3 = self.drug1_residual3_fc(x1)
#         x1 = self.drug1_conv3(x1, edge_index1)
#         x1 = self.relu(x1 + residual3)  # Residual connection
#
#         x1 = gmp(x1, batch1)  # global max pooling
#         x1 = self.relu(self.drug1_fc_g1(x1))
#         x1 = self.dropout(x1)
#         x1 = self.drug1_fc_g2(x1)
#         x1 = self.dropout(x1)
#
#         # deal drug2
#         residual4 = self.drug2_residual1_fc(x2)
#         x2 = self.drug2_conv1(x2, edge_index2)
#         x2 = self.relu(x2 + residual4)  # Residual connection
#
#         residual5 = self.drug2_residual2_fc(x2)
#         x2 = self.drug2_conv2(x2, edge_index2)
#         x2 = self.relu(x2 + residual5)  # Residual connection
#
#         residual6 = self.drug2_residual3_fc(x2)
#         x2 = self.drug2_conv3(x2, edge_index2)
#         x2 = self.relu(x2 + residual6)  # Residual connection
#
#         x2 = gmp(x2, batch2)  # global max pooling
#         x2 = self.relu(self.drug2_fc_g1(x2))
#         x2 = self.dropout(x2)
#         x2 = self.drug2_fc_g2(x2)
#         x2 = self.dropout(x2)
#
#         # deal cell
#         cell_vector = F.normalize(cell, 2, 1)
#         cell_vector = self.reduction(cell_vector)
#
#         # concat
#         xc = torch.cat((x1, x2, cell_vector), 1)
#         xc = self.fc1(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         xc = self.fc2(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         out = self.out(xc)
#         return out


# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import TransformerConv, global_max_pool as gmp
# from torch_geometric.data import Data
#
# class GraphTransformerNet(torch.nn.Module):
#     def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=954,
#                  output_dim=128, dropout=0.2, heads=2):
#         super(GraphTransformerNet, self).__init__()
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#
#         # SMILES1 graph branch
#         self.n_output = n_output
#         self.drug1_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug1_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug1_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug1_conv4 = TransformerConv(num_features_xd * 4 * heads, num_features_xd * 8, heads=heads)
#         self.drug1_fc_g1 = torch.nn.Linear(num_features_xd * 8 * heads, num_features_xd * 4)
#         self.drug1_fc_g2 = torch.nn.Linear(num_features_xd * 4, output_dim)
#
#         # Linear layers to match dimensions for residual connections
#         self.drug1_residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * heads)
#         self.drug1_residual2_fc = torch.nn.Linear(num_features_xd * heads, num_features_xd * 2 * heads)
#         self.drug1_residual3_fc = torch.nn.Linear(num_features_xd * 2 * heads, num_features_xd * 4 * heads)
#         self.drug1_residual4_fc = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 8 * heads)
#
#         # SMILES2 graph branch
#         self.drug2_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug2_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug2_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug2_conv4 = TransformerConv(num_features_xd * 4 * heads, num_features_xd * 8, heads=heads)
#         self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 8 * heads, num_features_xd * 4)
#         self.drug2_fc_g2 = torch.nn.Linear(num_features_xd * 4, output_dim)
#
#         # Linear layers to match dimensions for residual connections
#         self.drug2_residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * heads)
#         self.drug2_residual2_fc = torch.nn.Linear(num_features_xd * heads, num_features_xd * 2 * heads)
#         self.drug2_residual3_fc = torch.nn.Linear(num_features_xd * 2 * heads, num_features_xd * 4 * heads)
#         self.drug2_residual4_fc = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 8 * heads)
#
#         # DL cell features
#         self.reduction = nn.Sequential(
#             nn.Linear(num_features_xt, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )
#
#         # combined layers
#         self.fc1 = nn.Linear(3 * output_dim, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.out = nn.Linear(128, self.n_output)
#
#     def forward(self, data1, data2):
#         x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
#         x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
#
#         # deal drug1
#         residual1 = self.drug1_residual1_fc(x1)
#         x1 = self.drug1_conv1(x1, edge_index1)
#         x1 = self.relu(x1 + residual1)  # Residual connection
#
#         residual2 = self.drug1_residual2_fc(x1)
#         x1 = self.drug1_conv2(x1, edge_index1)
#         x1 = self.relu(x1 + residual2)  # Residual connection
#
#         residual3 = self.drug1_residual3_fc(x1)
#         x1 = self.drug1_conv3(x1, edge_index1)
#         x1 = self.relu(x1 + residual3)  # Residual connection
#
#         residual4 = self.drug1_residual4_fc(x1)
#         x1 = self.drug1_conv4(x1, edge_index1)
#         x1 = self.relu(x1 + residual4)  # Residual connection
#
#         x1 = gmp(x1, batch1)  # global max pooling
#         x1 = self.relu(self.drug1_fc_g1(x1))
#         x1 = self.dropout(x1)
#         x1 = self.drug1_fc_g2(x1)
#         x1 = self.dropout(x1)
#
#         # deal drug2
#         residual4 = self.drug2_residual1_fc(x2)
#         x2 = self.drug2_conv1(x2, edge_index2)
#         x2 = self.relu(x2 + residual4)  # Residual connection
#
#         residual5 = self.drug2_residual2_fc(x2)
#         x2 = self.drug2_conv2(x2, edge_index2)
#         x2 = self.relu(x2 + residual5)  # Residual connection
#
#         residual6 = self.drug2_residual3_fc(x2)
#         x2 = self.drug2_conv3(x2, edge_index2)
#         x2 = self.relu(x2 + residual6)  # Residual connection
#
#         residual7 = self.drug2_residual4_fc(x2)
#         x2 = self.drug2_conv4(x2, edge_index2)
#         x2 = self.relu(x2 + residual7)  # Residual connection
#
#         x2 = gmp(x2, batch2)  # global max pooling
#         x2 = self.relu(self.drug2_fc_g1(x2))
#         x2 = self.dropout(x2)
#         x2 = self.drug2_fc_g2(x2)
#         x2 = self.dropout(x2)
#
#         # deal cell
#         cell_vector = F.normalize(cell, 2, 1)
#         cell_vector = self.reduction(cell_vector)
#
#         # concat
#         xc = torch.cat((x1, x2, cell_vector), 1)
#         xc = self.fc1(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         xc = self.fc2(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         out = self.out(xc)
#         return out

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import TransformerConv, global_max_pool as gmp
# import torch.nn as nn
#
# class GraphTransformerNet(torch.nn.Module):
#     def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=954,
#                  output_dim=128, dropout=0.2, heads=2):
#         super(GraphTransformerNet, self).__init__()
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#
#         # SMILES1 graph branch
#         self.n_output = n_output
#         self.drug1_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug1_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug1_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug1_conv4 = TransformerConv(num_features_xd * 4 * heads, num_features_xd * 8, heads=heads)
#         self.drug1_conv5 = TransformerConv(num_features_xd * 8 * heads, num_features_xd * 8, heads=heads)
#         self.drug1_conv6 = TransformerConv(num_features_xd * 8 * heads, num_features_xd * 4, heads=heads)
#         self.drug1_fc_g1 = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 2)
#         self.drug1_fc_g2 = torch.nn.Linear(num_features_xd * 2, output_dim)
#
#         # Linear layers to match dimensions for residual connections
#         self.drug1_residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * heads)
#         self.drug1_residual2_fc = torch.nn.Linear(num_features_xd * heads, num_features_xd * 2 * heads)
#         self.drug1_residual3_fc = torch.nn.Linear(num_features_xd * 2 * heads, num_features_xd * 4 * heads)
#         self.drug1_residual4_fc = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 8 * heads)
#         self.drug1_residual5_fc = torch.nn.Linear(num_features_xd * 8 * heads, num_features_xd * 8 * heads)
#         self.drug1_residual6_fc = torch.nn.Linear(num_features_xd * 8 * heads, num_features_xd * 4 * heads)
#
#         # SMILES2 graph branch
#         self.drug2_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug2_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug2_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug2_conv4 = TransformerConv(num_features_xd * 4 * heads, num_features_xd * 8, heads=heads)
#         self.drug2_conv5 = TransformerConv(num_features_xd * 8 * heads, num_features_xd * 8, heads=heads)
#         self.drug2_conv6 = TransformerConv(num_features_xd * 8 * heads, num_features_xd * 4, heads=heads)
#         self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 2)
#         self.drug2_fc_g2 = torch.nn.Linear(num_features_xd * 2, output_dim)
#
#         # Linear layers to match dimensions for residual connections
#         self.drug2_residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * heads)
#         self.drug2_residual2_fc = torch.nn.Linear(num_features_xd * heads, num_features_xd * 2 * heads)
#         self.drug2_residual3_fc = torch.nn.Linear(num_features_xd * 2 * heads, num_features_xd * 4 * heads)
#         self.drug2_residual4_fc = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 8 * heads)
#         self.drug2_residual5_fc = torch.nn.Linear(num_features_xd * 8 * heads, num_features_xd * 8 * heads)
#         self.drug2_residual6_fc = torch.nn.Linear(num_features_xd * 8 * heads, num_features_xd * 4 * heads)
#
#         # DL cell features
#         self.reduction = nn.Sequential(
#             nn.Linear(num_features_xt, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )
#
#         # combined layers
#         self.fc1 = nn.Linear(3 * output_dim, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.out = nn.Linear(128, self.n_output)
#
#     def forward(self, data1, data2):
#         x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
#         x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
#
#         # deal drug1
#         residual1 = self.drug1_residual1_fc(x1)
#         x1 = self.drug1_conv1(x1, edge_index1)
#         x1 = self.relu(x1 + residual1)  # Residual connection
#
#         residual2 = self.drug1_residual2_fc(x1)
#         x1 = self.drug1_conv2(x1, edge_index1)
#         x1 = self.relu(x1 + residual2)  # Residual connection
#
#         residual3 = self.drug1_residual3_fc(x1)
#         x1 = self.drug1_conv3(x1, edge_index1)
#         x1 = self.relu(x1 + residual3)  # Residual connection
#
#         residual4 = self.drug1_residual4_fc(x1)
#         x1 = self.drug1_conv4(x1, edge_index1)
#         x1 = self.relu(x1 + residual4)  # Residual connection
#
#         residual5 = self.drug1_residual5_fc(x1)
#         x1 = self.drug1_conv5(x1, edge_index1)
#         x1 = self.relu(x1 + residual5)  # Residual connection
#
#         residual6 = self.drug1_residual6_fc(x1)
#         x1 = self.drug1_conv6(x1, edge_index1)
#         x1 = self.relu(x1 + residual6)  # Residual connection
#
#         x1 = gmp(x1, batch1)  # global max pooling
#         x1 = self.relu(self.drug1_fc_g1(x1))
#         x1 = self.dropout(x1)
#         x1 = self.drug1_fc_g2(x1)
#         x1 = self.dropout(x1)
#
#         # deal drug2
#         residual1 = self.drug2_residual1_fc(x2)
#         x2 = self.drug2_conv1(x2, edge_index2)
#         x2 = self.relu(x2 + residual1)  # Residual connection
#
#         residual2 = self.drug2_residual2_fc(x2)
#         x2 = self.drug2_conv2(x2, edge_index2)
#         x2 = self.relu(x2 + residual2)  # Residual connection
#
#         residual3 = self.drug2_residual3_fc(x2)
#         x2 = self.drug2_conv3(x2, edge_index2)
#         x2 = self.relu(x2 + residual3)  # Residual connection
#
#         residual4 = self.drug2_residual4_fc(x2)
#         x2 = self.drug2_conv4(x2, edge_index2)
#         x2 = self.relu(x2 + residual4)  # Residual connection
#
#         residual5 = self.drug2_residual5_fc(x2)
#         x2 = self.drug2_conv5(x2, edge_index2)
#         x2 = self.relu(x2 + residual5)  # Residual connection
#
#         residual6 = self.drug2_residual6_fc(x2)
#         x2 = self.drug2_conv6(x2, edge_index2)
#         x2 = self.relu(x2 + residual6)  # Residual connection
#
#         x2 = gmp(x2, batch2)  # global max pooling
#         x2 = self.relu(self.drug2_fc_g1(x2))
#         x2 = self.dropout(x2)
#         x2 = self.drug2_fc_g2(x2)
#         x2 = self.dropout(x2)
#
#         # deal cell
#         cell_vector = F.normalize(cell, 2, 1)
#         cell_vector = self.reduction(cell_vector)
#
#         # concat
#         xc = torch.cat((x1, x2, cell_vector), 1)
#         xc = self.fc1(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         xc = self.fc2(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         out = self.out(xc)
#         return out


# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import TransformerConv, global_max_pool as gmp
# from torch_geometric.data import Data
#
# class GraphTransformerNet(torch.nn.Module):
#     def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=954,
#                  output_dim=128, dropout=0.2, heads=2):
#         super(GraphTransformerNet, self).__init__()
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#
#         # SMILES1 graph branch
#         self.n_output = n_output
#         self.drug1_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug1_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug1_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug1_conv4 = TransformerConv(num_features_xd * 4 * heads, num_features_xd * 8, heads=heads)
#         self.drug1_fc_g1 = torch.nn.Linear(num_features_xd * 8 * heads, num_features_xd * 4)
#         self.drug1_fc_g2 = torch.nn.Linear(num_features_xd * 4, output_dim)
#
#         # Linear layers to match dimensions for residual connections
#         self.drug1_residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * heads)
#         self.drug1_residual2_fc = torch.nn.Linear(num_features_xd * heads, num_features_xd * 2 * heads)
#         self.drug1_residual3_fc = torch.nn.Linear(num_features_xd * 2 * heads, num_features_xd * 4 * heads)
#         self.drug1_residual4_fc = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 8 * heads)
#
#         # SMILES2 graph branch
#         self.drug2_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug2_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug2_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug2_conv4 = TransformerConv(num_features_xd * 4 * heads, num_features_xd * 8, heads=heads)
#         self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 8 * heads, num_features_xd * 4)
#         self.drug2_fc_g2 = torch.nn.Linear(num_features_xd * 4, output_dim)
#
#         # Linear layers to match dimensions for residual connections
#         self.drug2_residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * heads)
#         self.drug2_residual2_fc = torch.nn.Linear(num_features_xd * heads, num_features_xd * 2 * heads)
#         self.drug2_residual3_fc = torch.nn.Linear(num_features_xd * 2 * heads, num_features_xd * 4 * heads)
#         self.drug2_residual4_fc = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 8 * heads)
#
#         # DL cell features
#         self.reduction = nn.Sequential(
#             nn.Linear(num_features_xt, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )
#
#         # combined layers
#         self.fc1 = nn.Linear(3 * output_dim, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.out = nn.Linear(128, self.n_output)
#
#     def forward(self, data1, data2):
#         x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
#         x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
#
#         # deal drug1
#         residual1 = self.drug1_residual1_fc(x1)
#         x1 = self.drug1_conv1(x1, edge_index1)
#         x1 = self.relu(x1 + residual1)  # Residual connection
#
#         residual2 = self.drug1_residual2_fc(x1)
#         x1 = self.drug1_conv2(x1, edge_index1)
#         x1 = self.relu(x1 + residual2)  # Residual connection
#
#         residual3 = self.drug1_residual3_fc(x1)
#         x1 = self.drug1_conv3(x1, edge_index1)
#         x1 = self.relu(x1 + residual3)  # Residual connection
#
#         residual4 = self.drug1_residual4_fc(x1)
#         x1 = self.drug1_conv4(x1, edge_index1)
#         x1 = self.relu(x1 + residual4)  # Residual connection
#
#         x1 = gmp(x1, batch1)  # global max pooling
#         x1 = self.relu(self.drug1_fc_g1(x1))
#         x1 = self.dropout(x1)
#         x1 = self.drug1_fc_g2(x1)
#         x1 = self.dropout(x1)
#
#         # deal drug2
#         residual4 = self.drug2_residual1_fc(x2)
#         x2 = self.drug2_conv1(x2, edge_index2)
#         x2 = self.relu(x2 + residual4)  # Residual connection
#
#         residual5 = self.drug2_residual2_fc(x2)
#         x2 = self.drug2_conv2(x2, edge_index2)
#         x2 = self.relu(x2 + residual5)  # Residual connection
#
#         residual6 = self.drug2_residual3_fc(x2)
#         x2 = self.drug2_conv3(x2, edge_index2)
#         x2 = self.relu(x2 + residual6)  # Residual connection
#
#         residual7 = self.drug2_residual4_fc(x2)
#         x2 = self.drug2_conv4(x2, edge_index2)
#         x2 = self.relu(x2 + residual7)  # Residual connection
#
#         x2 = gmp(x2, batch2)  # global max pooling
#         x2 = self.relu(self.drug2_fc_g1(x2))
#         x2 = self.dropout(x2)
#         x2 = self.drug2_fc_g2(x2)
#         x2 = self.dropout(x2)
#
#         # deal cell
#         cell_vector = F.normalize(cell, 2, 1)
#         cell_vector = self.reduction(cell_vector)
#
#         # concat
#         xc = torch.cat((x1, x2, cell_vector), 1)
#         xc = self.fc1(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         xc = self.fc2(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         out = self.out(xc)
#         return out


import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_max_pool as gmp
from torch_geometric.data import Data
import torch.nn as nn


class CellFeatureExtractorBiLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super(CellFeatureExtractorBiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.bilstm(x)  # x should have shape (batch_size, seq_len, input_dim)
        x = self.relu(x[:, -1, :])
        x = self.fc(x)
        return x


class GraphTransformerNet(torch.nn.Module):
    def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=954,
                 output_dim=128, dropout=0.2, heads=6, hidden_dim=128, num_layers=2):
        super(GraphTransformerNet, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # SMILES1 graph branch
        self.n_output = n_output
        self.drug1_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
        self.drug1_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
        self.drug1_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
        self.drug1_conv4 = TransformerConv(num_features_xd * 4 * heads, num_features_xd * 8, heads=heads)
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd * 8 * heads, num_features_xd * 4)
        self.drug1_fc_g2 = torch.nn.Linear(num_features_xd * 4, output_dim)

        # Linear layers to match dimensions for residual connections
        self.drug1_residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * heads)
        self.drug1_residual2_fc = torch.nn.Linear(num_features_xd * heads, num_features_xd * 2 * heads)
        self.drug1_residual3_fc = torch.nn.Linear(num_features_xd * 2 * heads, num_features_xd * 4 * heads)
        self.drug1_residual4_fc = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 8 * heads)

        # SMILES2 graph branch
        self.drug2_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
        self.drug2_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
        self.drug2_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
        self.drug2_conv4 = TransformerConv(num_features_xd * 4 * heads, num_features_xd * 8, heads=heads)
        self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 8 * heads, num_features_xd * 4)
        self.drug2_fc_g2 = torch.nn.Linear(num_features_xd * 4, output_dim)

        # Linear layers to match dimensions for residual connections
        self.drug2_residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * heads)
        self.drug2_residual2_fc = torch.nn.Linear(num_features_xd * heads, num_features_xd * 2 * heads)
        self.drug2_residual3_fc = torch.nn.Linear(num_features_xd * 2 * heads, num_features_xd * 4 * heads)
        self.drug2_residual4_fc = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 8 * heads)

        # DL cell features with BiLSTM
        self.reduction = CellFeatureExtractorBiLSTM(num_features_xt, output_dim, hidden_dim, num_layers, dropout)

        # combined layers
        self.fc1 = nn.Linear(3 * output_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, self.n_output)

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # deal drug1
        residual1 = self.drug1_residual1_fc(x1)
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.relu(x1 + residual1)  # Residual connection

        residual2 = self.drug1_residual2_fc(x1)
        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.relu(x1 + residual2)  # Residual connection

        residual3 = self.drug1_residual3_fc(x1)
        x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.relu(x1 + residual3)  # Residual connection

        residual4 = self.drug1_residual4_fc(x1)
        x1 = self.drug1_conv4(x1, edge_index1)
        x1 = self.relu(x1 + residual4)  # Residual connection

        x1 = gmp(x1, batch1)  # global max pooling
        x1 = self.relu(self.drug1_fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.drug1_fc_g2(x1)
        x1 = self.dropout(x1)

        # deal drug2
        residual4 = self.drug2_residual1_fc(x2)
        x2 = self.drug2_conv1(x2, edge_index2)
        x2 = self.relu(x2 + residual4)  # Residual connection

        residual5 = self.drug2_residual2_fc(x2)
        x2 = self.drug2_conv2(x2, edge_index2)
        x2 = self.relu(x2 + residual5)  # Residual connection

        residual6 = self.drug2_residual3_fc(x2)
        x2 = self.drug2_conv3(x2, edge_index2)
        x2 = self.relu(x2 + residual6)  # Residual connection

        residual7 = self.drug2_residual4_fc(x2)
        x2 = self.drug2_conv4(x2, edge_index2)
        x2 = self.relu(x2 + residual7)  # Residual connection

        x2 = gmp(x2, batch2)  # global max pooling
        x2 = self.relu(self.drug2_fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.drug2_fc_g2(x2)
        x2 = self.dropout(x2)

        # deal cell with BiLSTM
        cell_vector = self.reduction(cell.unsqueeze(1))  # Ensure input shape is (batch_size, seq_len, input_dim)

        # Ensure cell_vector has the correct number of dimensions
        if len(cell_vector.size()) == 1:
            cell_vector = cell_vector.unsqueeze(0)

        # Ensure cell_vector matches the batch size
        if cell_vector.size(0) != x1.size(0):
            cell_vector = cell_vector.expand(x1.size(0), -1)

        # concat
        xc = torch.cat((x1, x2, cell_vector), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out



# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import TransformerConv, global_max_pool as gmp
# from torch_geometric.data import Data
#
# class GraphTransformerNet(torch.nn.Module):
#     def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=954,
#                  output_dim=128, dropout=0.2, heads=2):
#         super(GraphTransformerNet, self).__init__()
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#
#         # SMILES1 graph branch
#         self.n_output = n_output
#         self.drug1_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug1_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug1_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug1_conv4 = TransformerConv(num_features_xd * 4 * heads, num_features_xd * 8, heads=heads)
#         self.drug1_fc_g1 = torch.nn.Linear(num_features_xd * 8 * heads, num_features_xd * 4)
#         self.drug1_fc_g2 = torch.nn.Linear(num_features_xd * 4, output_dim)
#
#         # Linear layers to match dimensions for residual connections
#         self.drug1_residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * heads)
#         self.drug1_residual2_fc = torch.nn.Linear(num_features_xd * heads, num_features_xd * 2 * heads)
#         self.drug1_residual3_fc = torch.nn.Linear(num_features_xd * 2 * heads, num_features_xd * 4 * heads)
#         self.drug1_residual4_fc = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 8 * heads)
#
#         # SMILES2 graph branch
#         self.drug2_conv1 = TransformerConv(num_features_xd, num_features_xd, heads=heads)
#         self.drug2_conv2 = TransformerConv(num_features_xd * heads, num_features_xd * 2, heads=heads)
#         self.drug2_conv3 = TransformerConv(num_features_xd * 2 * heads, num_features_xd * 4, heads=heads)
#         self.drug2_conv4 = TransformerConv(num_features_xd * 4 * heads, num_features_xd * 8, heads=heads)
#         self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 8 * heads, num_features_xd * 4)
#         self.drug2_fc_g2 = torch.nn.Linear(num_features_xd * 4, output_dim)
#
#         # Linear layers to match dimensions for residual connections
#         self.drug2_residual1_fc = torch.nn.Linear(num_features_xd, num_features_xd * heads)
#         self.drug2_residual2_fc = torch.nn.Linear(num_features_xd * heads, num_features_xd * 2 * heads)
#         self.drug2_residual3_fc = torch.nn.Linear(num_features_xd * 2 * heads, num_features_xd * 4 * heads)
#         self.drug2_residual4_fc = torch.nn.Linear(num_features_xd * 4 * heads, num_features_xd * 8 * heads)
#
#         # DL cell features
#         self.reduction = nn.Sequential(
#             nn.Linear(num_features_xt, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )
#
#         # combined layers
#         self.fc1 = nn.Linear(3 * output_dim, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.out = nn.Linear(128, self.n_output)
#
#     def forward(self, data1, data2):
#         x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
#         x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
#
#         # deal drug1
#         residual1 = self.drug1_residual1_fc(x1)
#         x1 = self.drug1_conv1(x1, edge_index1)
#         x1 = self.relu(x1 + residual1)  # Residual connection
#
#         residual2 = self.drug1_residual2_fc(x1)
#         x1 = self.drug1_conv2(x1, edge_index1)
#         x1 = self.relu(x1 + residual2)  # Residual connection
#
#         residual3 = self.drug1_residual3_fc(x1)
#         x1 = self.drug1_conv3(x1, edge_index1)
#         x1 = self.relu(x1 + residual3)  # Residual connection
#
#         residual4 = self.drug1_residual4_fc(x1)
#         x1 = self.drug1_conv4(x1, edge_index1)
#         x1 = self.relu(x1 + residual4)  # Residual connection
#
#         x1 = gmp(x1, batch1)  # global max pooling
#         x1 = self.relu(self.drug1_fc_g1(x1))
#         x1 = self.dropout(x1)
#         x1 = self.drug1_fc_g2(x1)
#         x1 = self.dropout(x1)
#
#         # deal drug2
#         residual4 = self.drug2_residual1_fc(x2)
#         x2 = self.drug2_conv1(x2, edge_index2)
#         x2 = self.relu(x2 + residual4)  # Residual connection
#
#         residual5 = self.drug2_residual2_fc(x2)
#         x2 = self.drug2_conv2(x2, edge_index2)
#         x2 = self.relu(x2 + residual5)  # Residual connection
#
#         residual6 = self.drug2_residual3_fc(x2)
#         x2 = self.drug2_conv3(x2, edge_index2)
#         x2 = self.relu(x2 + residual6)  # Residual connection
#
#         residual7 = self.drug2_residual4_fc(x2)
#         x2 = self.drug2_conv4(x2, edge_index2)
#         x2 = self.relu(x2 + residual7)  # Residual connection
#
#         x2 = gmp(x2, batch2)  # global max pooling
#         x2 = self.relu(self.drug2_fc_g1(x2))
#         x2 = self.dropout(x2)
#         x2 = self.drug2_fc_g2(x2)
#         x2 = self.dropout(x2)
#
#         # deal cell
#         cell_vector = F.normalize(cell, 2, 1)
#         cell_vector = self.reduction(cell_vector)
#
#         # concat
#         xc = torch.cat((x1, x2, cell_vector), 1)
#         xc = self.fc1(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         xc = self.fc2(xc)
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         out = self.out(xc)
#         return out
