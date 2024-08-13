import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, activation: nn.Module = nn.SiLU(), dtype: torch.dtype = torch.float32):
        super(MLPNetwork, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim, dtype=dtype))
            layers.append(activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim, dtype=dtype))
        self.network = nn.Sequential(*layers)
        
        # Apply custom initialization
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # You can use different initializations based on your preference
            # For Xavier Normal Initialization
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            # For Kaiming Normal Initialization (use this if you're using ReLU or similar activations)
            # nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            # if module.bias is not None:
                # nn.init.constant_(module.bias, 0)
