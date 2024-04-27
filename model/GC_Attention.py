import torch
import torch.nn as nn
from model.TCN_Block import TCNBlock, TCNResidualBlock
from model.GCN_Block import GCNormalBlock
from model.Multihead_Attention import MultiheadAttentionWithChannelEmbedding, MultiheadAttentionWithNormalEmbedding
from model.Residual_Block import ResidualBlock
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GC_Attention_NormalEmbedding(nn.Module):
    def __init__(self, in_features, out_features, adjacency_matrix, degree_matrix, num_heads, embedding_dim):
        super(GC_Attention_NormalEmbedding, self).__init__()
        self.GCNBlock1 = GCNormalBlock(in_features=in_features, out_features=in_features, adjacency_matrix=adjacency_matrix, degree_matrix=degree_matrix).to(device)
        self.GCNBlock2 = GCNormalBlock(in_features=in_features, out_features=in_features, adjacency_matrix=adjacency_matrix, degree_matrix=degree_matrix).to(device)
        self.Attention = MultiheadAttentionWithNormalEmbedding(input_dim=in_features, output_dim=out_features, num_heads=num_heads, embedding_dim=embedding_dim).to(device)
        self.ResidualBlock = ResidualBlock(in_features=out_features).to(device)

    def forward(self, node_features, X_s):
        h_gc_1 = self.GCNBlock1(node_features)
        h_gc_2 = self.GCNBlock2(h_gc_1)
        Attention_value = self.Attention(h_gc_2, X_s)
        result = self.ResidualBlock(Attention_value)
        return result