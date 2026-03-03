import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_, out_):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_, out_, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_),
            nn.ReLU(),
            nn.Conv2d(out_, out_, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_),
            nn.ReLU()
        )

    def forward(self, input_: torch.Tensor):
        return self.double_conv(input_)

class ResidualBlock(nn.Module):
    def __init__(self, in_):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_, in_, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(in_),
            nn.ReLU(),
            nn.Conv2d(in_, in_, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(in_)
        )
        self.relu = nn.ReLU()

    def forward(self, input_: torch.Tensor):
        return self.relu(input_ + self.block(input_))


class Trunk(nn.Module):
    def __init__(self):
        super(Trunk, self).__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(10)]
        )

    def forward(self, board_info: torch.Tensor):
        return self.blocks(board_info)


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.input_ = DoubleConv(84, 64)
        
        self.trunk = Trunk()
        
        self.policy = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(start_dim = 1),
            nn.Linear(32 * 8 * 8, 64 * 64 + 88) 
        )
        
        self.value = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(start_dim = 1),
            nn.Linear(32 * 8 * 8, 1),
            nn.Tanh()
        )

    def forward(self, board_info: torch.Tensor):
        result = self.input_(board_info)
        result = self.trunk(result)
        return self.policy(result), self.value(result)