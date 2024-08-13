import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, activation: nn.Module = nn.SiLU(), dtype: torch.dtype = torch.float64):
        super(MLPNetwork, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim, dtype=dtype))
            layers.append(activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim, dtype=dtype))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
