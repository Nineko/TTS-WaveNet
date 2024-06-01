import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=2):
        super(DilatedCausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)

    def forward(self, x):
        # Apply convolution directly with padding
        return self.conv(x)[:, :, :-self.padding]

class WaveNetBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super(WaveNetBlock, self).__init__()
        self.dilated_conv = DilatedCausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.dilated_gate_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.condition_conv = nn.Conv1d(80, residual_channels, 1)  # 假設 80 個 Mel 頻帶
        self.condition_gate_conv = nn.Conv1d(80, residual_channels, 1)  # 假設 80 個 Mel 頻帶
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(self, x, condition):
        
        h = self.dilated_conv(x)
        h_cond = self.condition_conv(condition)
        h = h + h_cond
        h_g = self.dilated_gate_conv(x)
        h_g_cond = self.condition_gate_conv(condition)
        h_g = h_g + h_g_cond
        out = torch.tanh(h) * torch.sigmoid(h_g)
        residual = self.residual_conv(out)
        skip = self.skip_conv(out)
        return (x + residual) * 0.70710678118, skip  # Scale residual by sqrt(0.5) to stabilize training

class WaveNet(nn.Module):
    def __init__(self, layers, blocks, residual_channels, skip_channels, kernel_size):
        super(WaveNet, self).__init__()
        self.input_conv = nn.Conv1d(256, residual_channels, 1)
        self.residual_blocks = nn.ModuleList()
        
        for b in range(blocks):
            for l in range(layers):
                dilation = 2 ** l
                self.residual_blocks.append(
                    WaveNetBlock(residual_channels, skip_channels, kernel_size, dilation))
        
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, 256, 1)
    
    def forward(self, x, condition):
        x = self.input_conv(x)
        skip_connections = []
        
        for block in self.residual_blocks:
            x, skip = block(x, condition)
            skip_connections.append(skip)
        
        x = sum(skip_connections)
        x = F.relu(self.output_conv1(x))
        x = self.output_conv2(x)
        return x

#model = WaveNet(layers=10, blocks=3, residual_channels=32, skip_channels=256, kernel_size=3).to('cuda')
#summary(model, input_size=[(256, 64000),(80,64000)])  