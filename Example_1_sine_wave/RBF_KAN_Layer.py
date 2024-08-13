import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class RadialBasisFunctionNetwork(nn.Module):
    def __init__(self, hidden_layers: List[int], min_grid: float = -1., max_grid: float = 1., grid_count: int = 5, apply_base_update: bool = False, activation: nn.Module = nn.SiLU(), grid_opt: bool = False, dtype: torch.dtype = torch.float32, noise_scale: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        # First layer with specified min_grid and max_grid
        self.layers.append(RadialBasisFunctionLayer(hidden_layers[0], hidden_layers[1], min_grid, max_grid, grid_count, apply_base_update, activation, grid_opt, dtype, noise_scale))
        # Subsequent layers with min_grid and max_grid set to -1 and 1
        for in_dim, out_dim in zip(hidden_layers[1:-1], hidden_layers[2:]):
            self.layers.append(RadialBasisFunctionLayer(in_dim, out_dim, -1., 1., grid_count, apply_base_update, activation, grid_opt, dtype, noise_scale))
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)  
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))  
        x = self.layers[-1](x)  
        return x

class RadialBasisFunctionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, min_grid: float = -1., max_grid: float = 1., grid_count: int = 5, apply_base_update: bool = False, activation: nn.Module = nn.SiLU(), grid_opt: bool = False, dtype: torch.dtype = torch.float32, noise_scale: float = 0.1):
        super().__init__()
        self.apply_base_update = apply_base_update
        self.activation = activation
        self.min_grid = min_grid
        self.max_grid = max_grid
        self.grid_count = grid_count
        self.grid = nn.Parameter(torch.linspace(min_grid, max_grid, grid_count, dtype=dtype), requires_grad=grid_opt)
        self.rbf_weight = nn.Parameter(torch.empty(in_features * grid_count, out_features, dtype=dtype))
        self.scale_base = nn.Parameter(torch.ones(out_features, dtype=dtype))
        self.scale_rbf = nn.Parameter(torch.ones(out_features, dtype=dtype))
        
        nn.init.xavier_normal_(self.rbf_weight)
        self.rbf_weight.data += torch.randn_like(self.rbf_weight) * noise_scale
        
        self.base_activation = nn.SiLU() if apply_base_update else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.rbf_weight.dtype)  
        x_unsqueezed = x.unsqueeze(-1)  
        rbf_basis = torch.exp(-((x_unsqueezed - self.grid) / ((self.max_grid - self.min_grid) / (self.grid_count - 1))) ** 2)  # Shape: (batch_size, in_features, grid_count)
        rbf_basis = rbf_basis.view(rbf_basis.size(0), -1)  # Shape: (batch_size, in_features * grid_count)
        
        rbf_output = torch.mm(rbf_basis, self.rbf_weight)  # Shape: (batch_size, out_features)
        
        if self.apply_base_update:
            # Compute base activation if required
            base_output = self.base_activation(x)
            base_output = base_output.mean(dim=-1, keepdim=True)  
            output = self.scale_base * base_output + self.scale_rbf * rbf_output
        else:
            output = self.scale_rbf * rbf_output
        
        return output