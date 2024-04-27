import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from preprocess.decompose import decompose_trend_remainder
from preprocess.fast_shapelet_kmeans import fast_shapelet_kmeans
from model.TGC_Attention import TGC_Attention
from utiles.dataloader import TimeSeriesDataset
from utiles.metrics import RMSELoss, MSE_loss, MAE_loss, r2_score
import matplotlib.pyplot as plt

# Hyperparameters for shapelet extraction
num_shapelets = 3  # Number of shapelets to extract for each row
shapelet_length = 10  # Length of each shapelet
num_clusters = 3 #Number of channel label

# Load edge features from od_matrix_filter.pt
edge_features = torch.load("D:/Research/2024-Bicycle usage prediction/TGC_Attention/od_matrix_filtered.pt")
# Calculate summary statistics for each row to generate node features
node_features = edge_features.sum(dim=1)  # Sum along the rows to get node features
# Print the shape of the node features tensor
print("Node features shape:", node_features.shape)
print("Edge features shape:", edge_features.shape)

# Sum along the third dimension to get the adjacency matrix
adj_matrix = torch.sum(edge_features, dim=2)

# Set diagonal elements to 1
torch.diagonal(adj_matrix).fill_(1)

# Set non-zero elements to 1
adj_matrix[adj_matrix != 0] = 1

# Calculate the degree matrix
degree_matrix = torch.sum(adj_matrix, dim=1)

# Convert degree matrix to a diagonal matrix
degree_matrix = torch.diag(degree_matrix)

# Calculate trend and remainder features for node_features
X_td, X_s = decompose_trend_remainder(node_features)

# Calculate h_td = (Degree matrix - Adjacency matrix) * X_td
h_td = torch.abs(torch.matmul(degree_matrix - adj_matrix, X_td))

# Get the cluster labels for each station
cluster_labels = fast_shapelet_kmeans(h_td, shapelet_length, num_shapelets, num_clusters)
label_list = torch.tensor(cluster_labels).cuda()

#Set cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters for model
num_nodes = edge_features.size(0)
kernel_size= 3
dilation1 = 1
dilation2 = 2
num_heads = 2
embedding_dim=18
in_length = 5  
out_length = 3

# Define number of epochs
num_epochs = 1
learning_rate = 0.001

model = TGC_Attention(num_nodes=num_nodes, in_features=in_length, out_features=out_length, 
                      kernel_size=kernel_size, dilation1=dilation1, dilation2=dilation2, 
                      adjacency_matrix=adj_matrix, degree_matrix=degree_matrix,
                      num_heads=num_heads, embedding_dim=embedding_dim)

edge_features = edge_features.cuda()
node_features = node_features.cuda()
X_s = X_s.cuda()

# Create dataset
dataset = TimeSeriesDataset(edge_features, node_features, X_s, node_features, in_length, out_length)

# Define the size for training set (80%)
train_size = int(0.5 * len(dataset))

# Create training and test subsets
train_subset = Subset(dataset, range(train_size))
test_subset = Subset(dataset, range(train_size, len(dataset)))
train_dataloader = DataLoader(train_subset, batch_size=None, shuffle=True)
test_dataloader = DataLoader(test_subset, batch_size=None, shuffle=False)

# Define the optimizer
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = RMSELoss()

# Initialize lists to store loss values
rmse_losses = []

# Training loop
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()
    
    # Initialize loss accumulator
    total_rmse_loss = 0.0
    
    # Iterate over batches
    for batch_idx, (edge_features_seq, node_features_seq, X_s_seq, target) in enumerate(train_dataloader):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model.forward(edge_features=edge_features_seq, node_features=node_features_seq, label_list=label_list, X_s=X_s_seq)
        
        # Calculate loss
        rmse_loss = loss_function(output, target)

        # Backward pass
        rmse_loss.backward()

        # Update weights
        optimizer.step()
        
        # Accumulate loss
        total_rmse_loss += rmse_loss.item()

        # print(f"Batch [{batch_idx}] RMSE Loss: {rmse_loss:.4f}")

    # Calculate average loss for the epoch
    avg_rmse_loss = total_rmse_loss / len(train_dataloader)

    # Store loss values
    rmse_losses.append(avg_rmse_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg RMSE Loss: {avg_rmse_loss:.4f}")


# Evaluation loop (not in the training loop)
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    total_rmse = 0.0
    total_mae = 0.0
    total_r2 = 0.0
    
    for edge_features_seq, node_features_seq, X_s_seq, target in test_dataloader:
        # Forward pass
        output = model(edge_features=edge_features_seq, node_features=node_features_seq, label_list=label_list, X_s=X_s_seq)
        
        # Calculate RMSE
        rmse = loss_function(output, target)
        total_rmse += rmse.item()
        
        # Calculate MAE
        mae = MAE_loss(output, target)
        total_mae += mae.item()

        #Calculate R2
        r2 = r2_score(output, target)
        total_r2 += r2.item()
    
    # Average RMSE and MAE over all batches
    avg_rmse = total_rmse / len(test_dataloader)
    avg_mae = total_mae / len(test_dataloader)
    avg_r2 = total_r2 / len(test_dataloader)


    print(f"Average RMSE on Test Set: {avg_rmse:.4f}")
    print(f"Average MAE on Test Set: {avg_mae:.4f}")
    print(f"Average R2 on Test Set: {avg_r2:.4f}")


plt.figure(figsize=(8,5),dpi=300)
plt.plot(range(1, num_epochs+1), rmse_losses, label='RMSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()