import torch
import torch.nn as nn
class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, input):
        input = input.squeeze()
        return input.div(torch.norm(input, dim=1).view(-1, 1))

    def __repr__(self):
        return self.__class__.__name__