import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TCNBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation1, dilation2):
        super(TCNBlock, self).__init__()
        self.dcc_conv_1 = nn.Conv1d(in_channels, in_channels, kernel_size, dilation=dilation1, padding=(kernel_size - 1) * dilation1).to(device)
        self.dcc_conv_2 = nn.Conv1d(in_channels, in_channels, kernel_size, dilation=dilation2, padding=(kernel_size - 1) * dilation2).to(device)
        self.relu = nn.ReLU().to(device)
        self.weight_norm = nn.utils.parametrizations.weight_norm(self.dcc_conv_2).to(device)

    def forward(self, x):
        input_size = x.size()
        # Apply the first convolutional layer
        out = self.dcc_conv_1(x)
        # Apply the second convolutional layer
        out = self.dcc_conv_2(out)
        # Apply weight normalization
        out = self.weight_norm(out)
        # Apply ReLU activation
        out = self.relu(out)
        # Permute the output tensor back to the original shape
        if out.size()[-1] != input_size[-1]:
            diff = input_size[-1] - out.size()[-1]
            padding = (diff, 0)
            out = F.pad(out, padding).cuda()
        return out

class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TCNResidualBlock, self).__init__()
        self.conv_res = nn.Conv1d(in_channels, out_channels, 1).to(device)
    def forward(self, x, TCN_out):
        out = self.conv_res(x)
        out = out + TCN_out
        out = F.softmax(out, dim=-1).cuda()
        return out