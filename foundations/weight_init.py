import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        sigma = math.sqrt(2 / (fan_in + fan_out))
        W = torch.randn(fan_out, fan_in)
        W *= sigma
        W = torch.round(W, decimals=4)
        return W.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        sigma = math.sqrt(2 / fan_in)
        W = torch.randn(fan_out, fan_in)
        W *= sigma
        W = torch.round(W, decimals=4)
        return W.tolist()
        

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # This problem is broken af, you need to init
        # x after you init all weight matrices
        torch.manual_seed(0)
        weights = []
        std = []
        fan_in = input_dim
        fan_out = hidden_dim

        for _ in range(num_layers):
            W = torch.randn(fan_out, fan_in)
            if init_type == 'xavier':
                sigma = math.sqrt(2 / (fan_in + fan_out))
                W *= sigma
            elif init_type == 'kaiming':
                sigma = math.sqrt(2 / fan_in)
                W *= sigma
            weights.append(W)

        x = torch.randn(input_dim)
        
        # Apply the nonlinear transformation
        for W in weights:
            x = W @ x
            x = torch.relu(x)
            std.append(round(x.std().item(), 2))

        return [round(sigma, 2) for sigma in std]
