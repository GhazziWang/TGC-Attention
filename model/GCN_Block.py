import torch
import torch.nn as nn
import torch.nn.functional as F
import utiles.utiles

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GCNBlock(nn.Module):
    def __init__(self, in_features, out_features, adjacency_matrix, degree_matrix):
        super(GCNBlock, self).__init__()
        self.adjacency_matrix = adjacency_matrix.cuda()
        self.degree_matrix = degree_matrix.cuda()
        self.degree_matrix = torch.diag(torch.sum(self.adjacency_matrix, dim=1)).to(device)
        self.degree_matrix_sqrt_inv = torch.sqrt(torch.inverse(self.degree_matrix)).to(device)
        self.linear = nn.Linear(in_features, out_features).to(device)

    def forward(self, x, Residual_out):
        # Normalize adjacency matrix
        normalized_adjacency_matrix = torch.matmul(torch.matmul(self.degree_matrix_sqrt_inv, self.adjacency_matrix), self.degree_matrix_sqrt_inv).cuda()
        TCN_normalized_adjacency_matrix = normalized_adjacency_matrix.unsqueeze(-1).expand(-1, -1, Residual_out.size(-1)).cuda()
        weight_matrix = TCN_normalized_adjacency_matrix * Residual_out
        
        # Perform graph convolution
        out = utiles.utiles.custom_matrix_multiply(weight_matrix, x)
        out = self.linear(out).cuda()
        out = F.relu(out)
        return out
    

class GCNormalBlock(nn.Module):
    def __init__(self, in_features, out_features, adjacency_matrix, degree_matrix):
        super(GCNormalBlock, self).__init__()
        self.adjacency_matrix = adjacency_matrix.cuda()
        self.degree_matrix = degree_matrix.cuda()
        self.degree_matrix = torch.diag(torch.sum(self.adjacency_matrix, dim=1)).to(device)
        self.degree_matrix_sqrt_inv = torch.sqrt(torch.inverse(self.degree_matrix)).to(device)
        self.linear = nn.Linear(in_features, out_features).to(device)

    def forward(self, x):
        # Normalize adjacency matrix
        normalized_adjacency_matrix = torch.matmul(torch.matmul(self.degree_matrix_sqrt_inv, self.adjacency_matrix), self.degree_matrix_sqrt_inv).cuda()
      
        # Perform graph convolution
        out = torch.matmul(normalized_adjacency_matrix, x)
        out = self.linear(out).cuda()
        out = F.relu(out)
        return out