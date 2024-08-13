import h5py
import numpy as np
import os
from MLPNet import MLPNetwork as MLPNET
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
c_train = torch.tensor(c_train, dtype=dtype, device=device)
y_train = torch.tensor(y_train, dtype=dtype, device=device)
c_test = torch.tensor(c_test, dtype=dtype, device=device)
y_test = torch.tensor(y_test, dtype=dtype, device=device)
x = torch.tensor(x, dtype=dtype, device=device).reshape(len(x),1)

print(f'c_train shape = {c_train.shape}')
print(f'y_train shape = {y_train.shape}')
print(f'c_test shape = {c_test.shape}')
print(f'y_test shape = {y_test.shape}')
print(f'x shape = {x.shape}')

print(f'c_train max = {c_train.max()}, c_train min = {c_train.min()}')
print(f'x max = {x.max()}, x min = {x.min()}')

input_dim_trunk = x.shape[1]
input_dim_branch = c_train.shape[1]
HD = 40  
hidden_layers_mlp = [350, 350] 
print(f'hidden_layers_mlp = {hidden_layers_mlp}')
activation = nn.Tanh()

# Hyperparameters
learning_rate = 1e-3
batch_size    = 1024
epochs        = 20_000
gamma         = 0.9
step_size     = 500
seed          = 84

c_train = c_train.clone().detach().requires_grad_(True).to(dtype)
y_train = y_train.clone().detach().requires_grad_(True).to(dtype)
c_test = c_test.clone().detach().requires_grad_(True).to(dtype)
y_test = y_test.clone().detach().requires_grad_(True).to(dtype)

class DeepONet(nn.Module):
    def __init__(self):
        super(DeepONet, self).__init__()
        self.trunk_net = MLPNET(input_dim=input_dim_trunk, hidden_layers=hidden_layers_mlp, activation=activation, output_dim=HD, dtype=dtype).to(device)
        self.branch_net = MLPNET(input_dim=input_dim_branch, hidden_layers=hidden_layers_mlp, activation=activation, output_dim=HD, dtype=dtype).to(device)

    def forward(self, x_branch, x_trunk):
        y_branch = self.branch_net(x_branch)
        y_trunk = self.trunk_net(x_trunk)
        
        if y_branch.shape[-1] != y_trunk.shape[-1]:
            raise AssertionError("Output sizes of branch net and trunk net do not match.")

        Y = torch.einsum('bk,nk->bn', y_branch, y_trunk)
        return Y

model = DeepONet()

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

y_pred_deeponet = np.concatenate(y_pred_list, axis=0)

np.save('y_pred_deeponet.npy', y_pred_deeponet)

print("Predictions generated and saved successfully.")
print(f"Shape of y_pred_deeponet: {y_pred_deeponet.shape}")

with open('training_results_deeponet.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])
    csv_writer.writerows(results)  

print("Training results saved in 'training_results_deeponet.csv'")