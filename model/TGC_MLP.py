import torch
import torch.nn as nn
from model.TCN_Block import TCNBlock, TCNResidualBlock
from model.GCN_Block import GCNBlock
from model.Multihead_Attention import MultiheadAttentionWithChannelEmbedding, MultiheadAttentionWithNormalEmbedding
from model.Residual_Block import ResidualBlock
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.residual = nn.Linear(input_size, output_size)

    def forward(self, x, x_s):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x)+ self.residual(x_s))
        return x

class TGC_MLP(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_size, out_features, kernel_size, dilation1, dilation2, adjacency_matrix, degree_matrix):
        super(TGC_MLP, self).__init__()
        self.TCN1 = TCNBlock(in_channels=num_nodes, kernel_size=kernel_size, dilation1=dilation1, dilation2=dilation2).to(device)
        self.TCN2 = TCNBlock(in_channels=num_nodes, kernel_size=kernel_size, dilation1=dilation1, dilation2=dilation2).to(device)
        self.TCNResidualBlock = TCNResidualBlock(in_channels=num_nodes, out_channels=num_nodes).to(device)
        self.GCNBlock1 = GCNBlock(in_features=in_features, out_features=in_features, adjacency_matrix=adjacency_matrix, degree_matrix=degree_matrix).to(device)
        self.GCNBlock2 = GCNBlock(in_features=in_features, out_features=in_features, adjacency_matrix=adjacency_matrix, degree_matrix=degree_matrix).to(device)
        self.MLP = MLP(input_size=in_features, hidden_size=hidden_size, output_size=out_features).to(device)
        self.ResidualBlock = ResidualBlock(in_features=out_features).to(device)

    def forward(self, edge_features, node_features, X_s):
        TCN_out_1 = self.TCN1.forward(edge_features)
        Residual_out_1 = self.TCNResidualBlock(edge_features, TCN_out_1)
        TCN_out_2 = self.TCN2.forward(TCN_out_1)
        Residual_out_2 = self.TCNResidualBlock(edge_features, TCN_out_2)
        h_gc_1 = self.GCNBlock1(node_features, Residual_out_1)
        h_gc_2 = self.GCNBlock2(h_gc_1, Residual_out_2)
        MLP_value = self.MLP(h_gc_2, X_s)
        result = self.ResidualBlock(MLP_value)
        return result
    

class D_linear(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(D_linear, self).__init__()
        self.MLP = MLP(input_size=in_features, hidden_size=hidden_size, output_size=out_features).to(device)
        self.ResidualBlock = ResidualBlock(in_features=out_features).to(device)
    def forward(self, x, X_s):
        MLP_value = self.MLP(x, X_s)
        result = self.ResidualBlock(MLP_value)
        return result