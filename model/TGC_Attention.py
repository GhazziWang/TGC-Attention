import torch
import torch.nn as nn
from model.TCN_Block import TCNBlock, TCNResidualBlock
from model.GCN_Block import GCNBlock
from model.Multihead_Attention import MultiheadAttentionWithChannelEmbedding, MultiheadAttentionWithNormalEmbedding
from model.Residual_Block import ResidualBlock
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TGC_Attention(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, kernel_size, dilation1, dilation2, adjacency_matrix, degree_matrix, num_heads, embedding_dim):
        super(TGC_Attention, self).__init__()
        self.TCN1 = TCNBlock(in_channels=num_nodes, kernel_size=kernel_size, dilation1=dilation1, dilation2=dilation2).to(device)
        self.TCN2 = TCNBlock(in_channels=num_nodes, kernel_size=kernel_size, dilation1=dilation1, dilation2=dilation2).to(device)
        self.TCNResidualBlock = TCNResidualBlock(in_channels=num_nodes, out_channels=num_nodes).to(device)
        self.GCNBlock1 = GCNBlock(in_features=in_features, out_features=in_features, adjacency_matrix=adjacency_matrix, degree_matrix=degree_matrix).to(device)
        self.GCNBlock2 = GCNBlock(in_features=in_features, out_features=in_features, adjacency_matrix=adjacency_matrix, degree_matrix=degree_matrix).to(device)
        self.Attention = MultiheadAttentionWithChannelEmbedding(input_dim=in_features, output_dim=out_features, num_heads=num_heads, embedding_dim=embedding_dim).to(device)
        self.ResidualBlock = ResidualBlock(in_features=out_features).to(device)

    def forward(self, edge_features, node_features, label_list, X_s):
        TCN_out_1 = self.TCN1.forward(edge_features)
        Residual_out_1 = self.TCNResidualBlock(edge_features, TCN_out_1)
        TCN_out_2 = self.TCN2.forward(TCN_out_1)
        Residual_out_2 = self.TCNResidualBlock(edge_features, TCN_out_2)
        h_gc_1 = self.GCNBlock1(node_features, Residual_out_1)
        h_gc_2 = self.GCNBlock2(h_gc_1, Residual_out_2)
        Attention_value = self.Attention(h_gc_2, label_list, X_s)
        result = self.ResidualBlock(Attention_value)
        return result

class TGC_Attention_NormalEmbedding(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, kernel_size, dilation1, dilation2, adjacency_matrix, degree_matrix, num_heads, embedding_dim):
        super(TGC_Attention_NormalEmbedding, self).__init__()
        self.TCN1 = TCNBlock(in_channels=num_nodes, kernel_size=kernel_size, dilation1=dilation1, dilation2=dilation2).to(device)
        self.TCN2 = TCNBlock(in_channels=num_nodes, kernel_size=kernel_size, dilation1=dilation1, dilation2=dilation2).to(device)
        self.TCNResidualBlock = TCNResidualBlock(in_channels=num_nodes, out_channels=num_nodes).to(device)
        self.GCNBlock1 = GCNBlock(in_features=in_features, out_features=in_features, adjacency_matrix=adjacency_matrix, degree_matrix=degree_matrix).to(device)
        self.GCNBlock2 = GCNBlock(in_features=in_features, out_features=in_features, adjacency_matrix=adjacency_matrix, degree_matrix=degree_matrix).to(device)
        self.Attention = MultiheadAttentionWithNormalEmbedding(input_dim=in_features, output_dim=out_features, num_heads=num_heads, embedding_dim=embedding_dim).to(device)
        self.ResidualBlock = ResidualBlock(in_features=out_features).to(device)

    def forward(self, edge_features, node_features, X_s):
        TCN_out_1 = self.TCN1.forward(edge_features)
        Residual_out_1 = self.TCNResidualBlock(edge_features, TCN_out_1)
        TCN_out_2 = self.TCN2.forward(TCN_out_1)
        Residual_out_2 = self.TCNResidualBlock(edge_features, TCN_out_2)
        h_gc_1 = self.GCNBlock1(node_features, Residual_out_1)
        h_gc_2 = self.GCNBlock2(h_gc_1, Residual_out_2)
        Attention_value = self.Attention(h_gc_2, X_s)
        result = self.ResidualBlock(Attention_value)
        return result

class iTransformerChannelEmbedding(nn.Module):
    def __init__(self, in_features, out_features, num_heads, embedding_dim):
        super(iTransformerChannelEmbedding, self).__init__()
        self.Attention = MultiheadAttentionWithChannelEmbedding(input_dim=in_features, output_dim=out_features, num_heads=num_heads, embedding_dim=embedding_dim).to(device)
        self.ResidualBlock = ResidualBlock(in_features=out_features).to(device)

    def forward(self, x, X_s, label_list):
        Attention_value = self.Attention(x, label_list, X_s)
        result = self.ResidualBlock(Attention_value)
        return result

  
class iTransformerNormalEmbedding(nn.Module):
    def __init__(self, in_features, out_features, num_heads, embedding_dim):
        super(iTransformerNormalEmbedding, self).__init__()
        self.Attention = MultiheadAttentionWithNormalEmbedding(input_dim=in_features, output_dim=out_features, num_heads=num_heads, embedding_dim=embedding_dim).to(device)
        self.ResidualBlock = ResidualBlock(in_features=out_features).to(device)

    def forward(self, x, X_s):
        Attention_value = self.Attention(x, X_s)
        result = self.ResidualBlock(Attention_value)
        return result