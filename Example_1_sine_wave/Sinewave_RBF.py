import h5py
import numpy as np
import os
from RBF_KAN_Layer import RadialBasisFunctionNetwork as RBFKAN
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch import autograd
from tqdm import tqdm
import jax.numpy as jnp
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import pandas as pd
import random
from statistics import mean, median
import csv

########################################################################################
# Loading the data
x = np.load('x.npy')
c_train = np.load('c_train.npy')
y_train = np.load('y_train.npy')
c_test = np.load('c_test.npy')
y_test = np.load('y_test.npy')

dtype = torch.float64  # or torch.float32 or torch.float64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
c_train = torch.tensor(c_train, device=device)
y_train = torch.tensor(y_train, device=device)
c_test = torch.tensor(c_test, device=device)
y_test = torch.tensor(y_test, device=device)
x = torch.tensor(x, device=device).reshape(len(x),1)

print(f'c_train shape = {c_train.shape}')
print(f'y_train shape = {y_train.shape}')
print(f'c_test shape = {c_test.shape}')
print(f'y_test shape = {y_test.shape}')
print(f'x shape = {x.shape}')

print(f'c_train max = {c_train.max()}, c_train min = {c_train.min()}')
print(f'x max = {x.max()}, x min = {x.min()}')

def define_kan_width(input_dim, W, repeat_hid, output_dim):
    width = [input_dim] + [W] * repeat_hid + [output_dim]
    return width

input_dim_trunk  = x.shape[1]
input_dim_branch = c_train.shape[1]
HD = 40
hid_trunk = 50
num_layer_trunk  = 2
hid_branch = 50
num_layer_branch = 2
trunk_min_grid = x.min()
trunk_max_grid = x.max()
branch_min_grid = c_train.min()
branch_max_grid = c_train.max()

width_trunk = define_kan_width(input_dim_trunk, hid_trunk, num_layer_trunk, HD)
width_branch = define_kan_width(input_dim_branch, hid_branch, num_layer_branch, HD)

print(f'hid dim = {hid_trunk}, num_layer = {num_layer_trunk}')
print(f'width_trunk = {width_trunk}, width_branch = {width_branch}')
grid_opt = True 
grid_count = 30
init_scale = 0.01
noise_scale = 0.01
print(f'init_scale = {init_scale}')

# Hyperparameters
learning_rate = 1e-2
batch_size    = 1024
epochs        = 20_000
gamma         = 0.9
step_size     = 500
seed          = 84

c_train = c_train.clone().detach().requires_grad_(True).to(dtype)
y_train = y_train.clone().detach().requires_grad_(True).to(dtype)
c_test = c_test.clone().detach().requires_grad_(True).to(dtype)
y_test = y_test.clone().detach().requires_grad_(True).to(dtype)

class DeepOKAN(nn.Module):
    def __init__(self):
        super(DeepOKAN, self).__init__()
        self.trunk_net  = RBFKAN(hidden_layers=width_trunk,  dtype=dtype, apply_base_update=False, noise_scale=noise_scale,
                                  min_grid=trunk_min_grid, max_grid=trunk_max_grid, grid_count=grid_count, grid_opt=grid_opt).to(device)
        self.branch_net = RBFKAN(hidden_layers=width_branch, dtype=dtype, apply_base_update=False, noise_scale=noise_scale,
                                  min_grid=branch_min_grid,  max_grid=branch_max_grid,  grid_count=grid_count, grid_opt=grid_opt).to(device)

    def forward(self, x_branch, x_trunk):
        y_branch = self.branch_net(x_branch)
        y_trunk = self.trunk_net(x_trunk)

        if y_branch.shape[-1] != y_trunk.shape[-1]:
            raise AssertionError("Output sizes of branch net and trunk net do not match.")

        Y = torch.einsum('bk,nk->bn', y_branch, y_trunk)

        return Y

model = DeepOKAN()

class RMSDLoss(nn.Module):
    def __init__(self):
        super(RMSDLoss, self).__init__()

    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError("Output and target must have the same shape for RMSD calculation.")
        squared_diffs = (output - target).pow(2)
        mean_squared_diffs = torch.mean(squared_diffs)
        rmsd_value = torch.sqrt(mean_squared_diffs)
        return rmsd_value
    
criterion = RMSDLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# L-BFGS optimizer with line search
# optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, line_search_fn='strong_wolfe')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(seed)

def test_model(model, criterion, c_test, y_test, x, batch_size):
    """Evaluates the model on the test dataset in batches."""
    model.eval()
    total_loss = 0
    num_batches = len(c_test) // batch_size + 1
    with torch.no_grad(): 
        for i in tqdm(range(0, len(c_test), batch_size), desc='Testing'):
            batch_loads = c_test[i:i + batch_size]
            batch_solutions = y_test[i:i + batch_size]
            outputs = model(batch_loads, x)
            loss = criterion(outputs, batch_solutions)
            total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return avg_loss

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
num_learnable_parameters = count_learnable_parameters(model)
print(f'The number of learnable parameters in the model: {num_learnable_parameters}')

import csv
import time
from tqdm import tqdm
import numpy as np
import torch

train_losses = []
test_losses = []
start_time = time.time()
mean_test_loss = 0

results = []

for epoch in range(epochs):
    model.train() 
    total_loss = 0
    progress_bar = tqdm(range(0, len(c_train), batch_size), desc=f'Epoch {epoch + 1}/{epochs}')
    for i in progress_bar:
        batch_loads = c_train[i:i + batch_size]
        batch_solutions = y_train[i:i + batch_size]

        def closure():
            optimizer.zero_grad()
            outputs = model(batch_loads, x)
            loss = criterion(outputs, batch_solutions)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        total_loss += loss.item()

        progress_bar.set_postfix({'Batch Loss': loss.item()})

    avg_loss = total_loss / (len(c_train) // batch_size + 1)
    train_losses.append(avg_loss)
    tqdm.write(f'End of Epoch {epoch + 1}, Average Loss: {avg_loss}')

    test_loss = test_model(model, criterion, c_test, y_test, x, batch_size)
    test_losses.append(test_loss)
    tqdm.write(f'Test Loss: {test_loss}')
    
    results.append([epoch + 1, avg_loss, test_loss])
    
    scheduler.step() 

end_time = time.time()
training_time = end_time - start_time
print('###################################################################################################')
print('###################################################################################################')
print('###################################################################################################')
print(f'Training time: {training_time:.2f} seconds')
print(f'The number of learnable parameters in the model: {num_learnable_parameters}')
print('###################################################################################################')
print('###################################################################################################')
print('###################################################################################################')

model.eval()
y_pred_list = []
batch_size = 100
with torch.no_grad():  
    for i in range(0, len(c_test), batch_size):
        batch_loads = c_test[i:i+batch_size].clone().detach().to(device)
        batch_pred = model(batch_loads, x)
        y_pred_list.append(batch_pred.cpu().numpy())

y_pred_deepokan = np.concatenate(y_pred_list, axis=0)

np.save('y_pred_deepokan.npy', y_pred_deepokan)

print("Predictions generated and saved successfully.")
print(f"Shape of y_pred_deepokan: {y_pred_deepokan.shape}")

with open('training_results_deepokan.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Test Loss']) 
    csv_writer.writerows(results)  

print("Training results saved in 'training_results_deepokan.csv'")