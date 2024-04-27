import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChannelEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(ChannelEmbedding, self).__init__()
        self.embedding_0 = nn.Linear(input_dim, embedding_dim).to(device) # Embedding for label 0
        self.embedding_1 = nn.Linear(input_dim, embedding_dim).to(device) # Embedding for label 1
        self.embedding_2 = nn.Linear(input_dim, embedding_dim).to(device) # Embedding for label 2
        
    def forward(self, input_tensor, label_list):
        embedded_tensors = []
        for label in label_list:
            selected_rows = input_tensor[label.item()].unsqueeze(0)  # Unsqueeze to add batch dimension
            if label == 0:
                embedded_tensor = self.embedding_0(selected_rows)
            elif label == 1:
                embedded_tensor = self.embedding_1(selected_rows)
            else:
                embedded_tensor = self.embedding_2(selected_rows)
            embedded_tensors.append(embedded_tensor)
        concatenated_tensor = torch.cat(embedded_tensors, dim=0)
        return concatenated_tensor
    

class NormalEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(NormalEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim).to(device) # Embedding for label 0
       
    def forward(self, input_tensor):
        result = self.embedding(input_tensor)
        return result