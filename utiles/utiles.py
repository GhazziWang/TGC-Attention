import torch

def custom_matrix_multiply(tensor1, tensor2):
    result = []
    # Iterate over the last dimension of tensor1
    for i in range(tensor1.shape[2]):
        # Extract the i-th slice from tensor1 and reshape it to (3, 3, 1)
        tensor1_slice = tensor1[:, :, i].squeeze(-1)
        # Multiply the slice with the corresponding column from tensor2
        product = torch.matmul(tensor1_slice, tensor2[:, i:i+1])
        # Append the result to the list
        result.append(product)
    
    # Concatenate along the last dimension to get the final result
    return torch.cat(result, dim=-1)    