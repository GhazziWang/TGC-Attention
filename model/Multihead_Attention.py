import torch
import torch.nn as nn
from model.Channel_Embedding import ChannelEmbedding, NormalEmbedding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiheadAttentionWithChannelEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, embedding_dim):
        super(MultiheadAttentionWithChannelEmbedding, self).__init__()
        self.channel_embedding = ChannelEmbedding(input_dim, embedding_dim).to(device)
        self.multihead_attention = nn.MultiheadAttention(embedding_dim, num_heads).to(device)
        self.out_linear_multihead = nn.Linear(embedding_dim, output_dim).to(device)
        self.out_linear_input = nn.Linear(input_dim, output_dim).to(device)
        self.out_linear_x_s = nn.Linear(input_dim, output_dim).to(device)
        self.norm = nn.LayerNorm(output_dim).to(device)
        
    def forward(self, input_tensor, label_list, x_s):
        embedded_tensor = self.channel_embedding(input_tensor, label_list)
        output_tensor, _ = self.multihead_attention(embedded_tensor, embedded_tensor, embedded_tensor)
        output_tensor = self.norm(self.out_linear_input(input_tensor) + self.out_linear_multihead(output_tensor) + self.out_linear_input(x_s)) 
        return output_tensor
    

class MultiheadAttentionWithNormalEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, embedding_dim):
        super(MultiheadAttentionWithNormalEmbedding, self).__init__()
        self.embedding = NormalEmbedding(input_dim, embedding_dim).to(device)
        self.multihead_attention = nn.MultiheadAttention(embedding_dim, num_heads).to(device)
        self.out_linear_multihead = nn.Linear(embedding_dim, output_dim).to(device)
        self.out_linear_input = nn.Linear(input_dim, output_dim).to(device)
        self.out_linear_x_s = nn.Linear(input_dim, output_dim).to(device)
        self.norm = nn.LayerNorm(output_dim).to(device)
        
    def forward(self, input_tensor, x_s):
        embedded_tensor = self.embedding(input_tensor)
        output_tensor, _ = self.multihead_attention(embedded_tensor, embedded_tensor, embedded_tensor)
        output_tensor = self.norm(self.out_linear_input(input_tensor) + self.out_linear_multihead(output_tensor) + self.out_linear_input(x_s)) 
        return output_tensor