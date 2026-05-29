import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from ttnn import *

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(Conv1dBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=dilation, bias=False)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv1dBlock(channels, channels, kernel_size, dilation)
        self.conv2 = Conv1dBlock(channels, channels, kernel_size, dilation)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        return out + residual * self.residual_scale

class LLVCEncoder(nn.Module):
    def __init__(self, num_mels=80, channels=512, in_channels=160, out_channels=512):
        super(LLVCEncoder, self).__init__()
        self.conv_pre = Conv1dBlock(in_channels, channels, kernel_size=3, dilation=1)
        self.conv_post = Conv1dBlock(channels, channels, kernel_size=1, dilation=1)
        self.conv_blocks = nn.ModuleList([
            ResidualBlock(channels, 3, 1),
            ResidualBlock(channels, 3, 3),
            ResidualBlock(channels, 3, 9),
            ResidualBlock(channels, 3, 27),
        ])

    def forward(self, x):
        x = self.conv_pre(x)
        for block in self.conv_blocks:
            x = block(x)
        x = self.conv_post(x)
        return x

class LLVCDecoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=160):
        self.conv_pre = Conv1dBlock(in_channels, in_channels, kernel_size=1, dilation=1)
        self.conv_post = Conv1dBlock(in_channels, out_channels, kernel, size=1, dilation=1)

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.conv_post(x)
        return x

class LLVCModel(nn.Module):
    def __init__(self):
        super(LLVCModel, self)it__init__()
        self.encoder = LLVCEncoder()
        self.decoder = LLVCDecoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def create_llvc_model():
    model = LLVCModel()
    return model

if __name__ == "__main__":
    model = create_llvc_model()
    print("LLVC Model initialized successfully")