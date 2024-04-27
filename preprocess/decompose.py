import torch
import torch.nn.functional as F

# Assuming node_features is your input tensor where each row represents a time series data for a station

# Function to calculate trend and remainder features for each row
def decompose_trend_remainder(node_features):
    trend_features = []
    remainder_features = []
    for row in node_features:
        # Pad the row to match the size after pooling
        row_padded = F.pad(row.unsqueeze(0), (1, 1), mode='reflect').squeeze(0)

        # Calculate trend features (X_td) using average pooling
        X_td = F.avg_pool1d(row_padded.unsqueeze(0), kernel_size=3, stride=1).squeeze(0)

        # Calculate remainder features (X_s)
        X_s = row - X_td
        
        trend_features.append(X_td)
        remainder_features.append(X_s)
    
    # Stack the trend and remainder features to form tensors
    trend_features = torch.stack(trend_features)
    remainder_features = torch.stack(remainder_features)
    
    return trend_features, remainder_features
