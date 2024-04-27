import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(in_features)
        self.ffn = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features)
        ).to(device)

    def forward(self, x):
        # Apply the feedforward network
        ffn_out = self.ffn(x)
        
        # Apply skip connection and layer normalization
        out = self.norm(ffn_out + x)
        
        # Apply ReLU activation
        out = F.relu(out)
        
        return out

