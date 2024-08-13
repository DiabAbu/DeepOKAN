import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class RadialBasisFunctionNetwork(nn.Module):
    def __init__(self, hidden_layers: List[int], min_grid: float = -1., max_grid: float = 1., grid_count: int = 5, apply_base_update: bool = False, activation: nn.Module = nn.SiLU(), grid_opt: bool = False, init_scale: float = 0.1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append(RadialBasisFunctionLayer(hidden_layers[0], hidden_layers[1], min_grid, max_grid, grid_count, apply_base_update, activation, grid_opt, init_scale, dtype))
        
        for in_dim, out_dim in zip(hidden_layers[1:-1], hidden_layers[2:]):
            self.layers.append(RadialBasisFunctionLayer(in_dim, out_dim, -1., 1., grid_count, apply_base_update, activation, grid_opt, init_scale, dtype))
        
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x

class RadialBasisFunctionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, min_grid: float = -1., max_grid: float = 1., grid_count: int = 5, apply_base_update: bool = False, activation: nn.Module = nn.SiLU(), grid_opt: bool = False, init_scale: float = 0.1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.apply_base_update = apply_base_update
        self.activation = activation
        self.min_grid = min_grid
        self.max_grid = max_grid
        self.grid_count = grid_count
        self.grid = nn.Parameter(torch.linspace(min_grid, max_grid, grid_count, dtype=dtype), requires_grad=grid_opt)
        self.rbf_weight = nn.Parameter(torch.randn(in_features * grid_count, out_features, dtype=dtype) * init_scale)
        self.base_linear = nn.Linear(in_features, out_features, dtype=dtype) if apply_base_update else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.rbf_weight.dtype)
        x_unsqueezed = x.unsqueeze(-1)  # Shape: (batch_size, in_features, 1)
        rbf_basis = torch.exp(-((x_unsqueezed - self.grid) / ((self.max_grid - self.min_grid) / (self.grid_count - 1))) ** 2)  # Shape: (batch_size, in_features, grid_count)
        rbf_basis = rbf_basis.view(rbf_basis.size(0), -1)  # Shape: (batch_size, in_features * grid_count)
        rbf_output = torch.einsum('bi,ij->bj', rbf_basis, self.rbf_weight)  # Shape: (batch_size, out_features)
        
        if self.apply_base_update:
            base_output = self.base_linear(self.activation(x))
            rbf_output += base_output
        
        return rbf_output