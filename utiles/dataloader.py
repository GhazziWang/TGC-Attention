import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, edge_features, node_features, X_s, target_features, in_length, out_length):
        self.edge_features = edge_features
        self.node_features = node_features
        self.X_s = X_s
        self.target_features = target_features
        self.in_length = in_length
        self.out_length = out_length

    def __len__(self):
        return self.node_features.size(1) - self.in_length - self.out_length

    def __getitem__(self, idx):
        end_idx = idx + self.in_length
        edge_seq = self.edge_features[:,:,idx:end_idx]
        node_seq = self.node_features[:,idx:end_idx]
        X_s_seq = self.X_s[:,idx:end_idx]
        target = self.target_features[:,end_idx:end_idx+self.out_length]  # Get the next 3 node features as target
        return edge_seq, node_seq, X_s_seq, target
    
class RNNTimeSeriesDataset(Dataset):
    def __init__(self, node_features, target_features, in_length, out_length):
        self.node_features = node_features
        self.target_features = target_features
        self.in_length = in_length
        self.out_length = out_length

    def __len__(self):
        return self.node_features.size(1) - self.in_length - self.out_length

    def __getitem__(self, idx):
        end_idx = idx + self.in_length
        node_seq = self.node_features[:,idx:end_idx]
        target = self.target_features[:,end_idx:end_idx+self.out_length]  # Get the next 3 node features as target
        return node_seq, target
    
class D_linearTimeSeriesDataset(Dataset):
    def __init__(self, node_features, target_features, X_s, in_length, out_length):
        self.node_features = node_features
        self.target_features = target_features
        self.x_s = X_s
        self.in_length = in_length
        self.out_length = out_length

    def __len__(self):
        return self.node_features.size(1) - self.in_length - self.out_length

    def __getitem__(self, idx):
        end_idx = idx + self.in_length
        node_seq = self.node_features[:,idx:end_idx]
        x_s_seq = self.x_s[:,idx:end_idx]
        target = self.target_features[:,end_idx:end_idx+self.out_length]  # Get the next 3 node features as target
        return node_seq, x_s_seq, target