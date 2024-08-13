import h5py
import numpy as np
import os
from KAN_RBF6 import RadialBasisFunctionNetwork as RBFKAN
from MLPNet import MLPNetwork as MLPNET
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch import autograd
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import scipy.io
import numpy.random as npr
import random
import time
import os

########################################################################################
HD                  = 5
hid_trunk           = 14
repeat_hid_trunk    = 1
hid_branch          = 14
repeat_hid_branch   = 1

init_scale          = 0.01
grid_count          = 5

hidden_layers_mlp   = [62] 
activation          = nn.Tanh()

learning_rate       = 1e-3
batch_size          = 64
epochs              = 1
gamma               = 0.5
step_size_scheduler = 1000
opt                 = "adam" # "adam" "lbfgs"

loss_vec_rbf        = []
loss_vec_mlp        = []
test_loss_vec_rbf   = []
test_loss_vec_mlp   = []
dtype               = torch.float64  # or torch.float32 or torch.float64
seeding_number      = 1

npr.seed(seeding_number); torch.manual_seed(seeding_number); random.seed(seeding_number)

########################################################################################
def param_and_concatenate_data(filepath):
    with h5py.File(filepath, 'r') as hdf:
        param_list = []
        disp_list = []

        param_group = hdf['params']
        for key in sorted(param_group.keys()):
            param_list.append(np.array(param_group[key]))

        disp_group = hdf['disp']
        for key in sorted(disp_group.keys()):
            disp_list.append(np.array(disp_group[key]))

        params = np.stack(param_list, axis=0)  
        disps = np.stack(disp_list, axis=0) 

        coordinates = np.array(hdf['coordinates'])

    return params, disps, coordinates

# ---------------------------------------------------------------------------------------
class L2NormLoss(nn.Module):
    def __init__(self):
        super(L2NormLoss, self).__init__()

    def forward(self, output, target):
        return torch.norm(output - target, p=2)
    
# ---------------------------------------------------------------------------------------
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

########################################################################################
filepath = 'C:/Users/pp2624/Desktop/Research/8. DeepOKANs/Analyses for Arxiv 2 - Orthotropic Elasticity/data_100percent.hdf5'
params, disp, coordinates = param_and_concatenate_data(filepath)
print('max min params')
print(params.max(), params.min())
print('max min coordinates')
print(coordinates.max(), coordinates.min())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device used is {device}')

params      = torch.tensor(params).to(dtype).to(device)
disp        = torch.tensor(disp).to(dtype).to(device)
coordinates = torch.tensor(coordinates).to(dtype).to(device)

print(f'params shape = {params.shape}')
print(f'disp shape = {disp.shape}')
print(f'coordinates shape = {coordinates.shape}')

def split_data(params, disp, test_ratio=0.2):
    """ Splits the data into training and testing sets. """
    num_data = params.shape[0]
    indices  = torch.randperm(num_data)
    split    = int(num_data * (1 - test_ratio))

    train_indices = indices[:split]
    test_indices  = indices[split:]

    train_params    = params[train_indices]
    train_true_ux   = disp[train_indices]
    test_params     = params[test_indices]
    test_true_ux    = disp[test_indices]

    return train_params, train_true_ux[:,:,0], test_params, test_true_ux[:,:,0]

train_params, train_true_ux, test_params, test_true_ux = split_data(params, disp)

########################################################################################
def define_kan_width(input_dim, hid, output_dim, repeat_hid):
    width = [input_dim]
    width += [hid] * repeat_hid
    width.append(output_dim)
    return width

width_trunk   = define_kan_width(coordinates.shape[1], hid_trunk, HD, repeat_hid_trunk)
width_branch  = define_kan_width(train_params.shape[1], hid_branch, HD, repeat_hid_branch)

print(f'width_trunk = {width_trunk}')
print(f'width_branch = {width_branch}')

train_params  = train_params.clone().detach().requires_grad_(True).to(dtype)
train_true_ux = train_true_ux.clone().detach().requires_grad_(True).to(dtype)
test_params   = test_params.clone().detach().requires_grad_(True).to(dtype)
test_true_ux  = test_true_ux.clone().detach().requires_grad_(True).to(dtype)


########################################################################################
class DeepONet_rbf(nn.Module):
    def __init__(self):
        super(DeepONet_rbf, self).__init__()
        self.trunk_net  = RBFKAN(hidden_layers=width_trunk,  dtype=dtype, apply_base_update=False, min_grid=0,  max_grid=1,  grid_count=grid_count, grid_opt=False, init_scale=init_scale).to(device)
        self.branch_net = RBFKAN(hidden_layers=width_branch, dtype=dtype, apply_base_update=False, min_grid=-1, max_grid=20, grid_count=grid_count, grid_opt=False, init_scale=init_scale).to(device)

    def forward(self, x_branch, x_trunk):
        y_branch = self.branch_net(x_branch)
        y_trunk  = self.trunk_net(x_trunk)
        if y_branch.shape[-1] != y_trunk.shape[-1]:
            raise AssertionError("Output sizes of branch net and trunk net do not match.")
        Y = torch.einsum('bi,ni->bn', y_branch, y_trunk)
        return Y

model_rbf = DeepONet_rbf()

########################################################################################
# Hyperparameters
criterion1      = nn.MSELoss()
criterion2      = L2NormLoss()
criterion3      = RMSDLoss()

if opt == "adam":
    optimizer_rbf = optim.Adam(model_rbf.parameters(), lr=learning_rate)
    scheduler_rbf = StepLR(optimizer_rbf, step_size=step_size_scheduler, gamma=gamma)
else:
    optimizer_rbf = torch.optim.LBFGS(model_rbf.parameters(), lr=1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

def test_model(model, criterion, test_params, test_disp, coordinates):
    """Evaluates the model on the test dataset."""
    model.eval()  
    with torch.no_grad():
        outputs = model(test_params, coordinates)
        loss = criterion(outputs, test_disp)
    return loss.item()

# --------------------------------------------------------------------------------------
start_train_time_rbf = time.time()
for epoch in range(epochs):
    total_loss = 0
    progress_bar = tqdm(range(0, len(train_params), batch_size), desc=f'Epoch {epoch + 1}/{epochs}')
    for i in progress_bar:
        batch_params = train_params[i:i + batch_size]
        batch_disp = train_true_ux[i:i + batch_size]
        def closure_rbf():
            optimizer_rbf.zero_grad()
            outputs = model_rbf(batch_params, coordinates)
            loss = criterion3(outputs, batch_disp)
            loss.backward()
            return loss
        loss = optimizer_rbf.step(closure_rbf)
        total_loss += loss.item()
        progress_bar.set_postfix({'Batch Loss': loss.item()})

    avg_loss_rbf = total_loss / (len(train_params) // batch_size)
    loss_vec_rbf.append(avg_loss_rbf)
    tqdm.write(f'End of Epoch {epoch + 1}, Average Loss: {avg_loss_rbf}')

    test_loss_rbf = test_model(model_rbf, criterion3, test_params, test_true_ux, coordinates)
    test_loss_vec_rbf.append(test_loss_rbf)
    tqdm.write(f'Test Loss: {test_loss_rbf}')
    
    # if (epoch + 1) % 1005 == 0:
    #     print("Heyyyy")
    #     with torch.no_grad():
    #         test_pred_ux_rbf    = model_rbf(test_params, coordinates).cpu().numpy()
    #         scipy.io.savemat('Results_Orthotropic_RBF_' + str(epoch + 1) + '.mat', {'test_pred_ux_rbf': test_pred_ux_rbf, 'loss_vec_rbf':loss_vec_rbf})

    if opt == "adam":
        scheduler_rbf.step()

end_training_time_rbf = time.time() - start_train_time_rbf

########################################################################################
########################################################################################
########################################################################################
class DeepONet_mlp(nn.Module):
    def __init__(self):
        super(DeepONet_mlp, self).__init__()
        self.trunk_net  = MLPNET(input_dim=2, hidden_layers=hidden_layers_mlp, activation=activation, output_dim=HD, dtype=dtype).to(device)
        self.branch_net = MLPNET(input_dim=6, hidden_layers=hidden_layers_mlp, activation=activation, output_dim=HD, dtype=dtype).to(device)

    def forward(self, x_branch, x_trunk):
        y_branch = self.branch_net(x_branch)
        y_trunk  = self.trunk_net(x_trunk)
        if y_branch.shape[-1] != y_trunk.shape[-1]:
            raise AssertionError("Output sizes of branch net and trunk net do not match.")
        Y = torch.einsum('bi,ni->bn', y_branch, y_trunk)
        return Y

model_mlp = DeepONet_mlp()

if opt == "adam":
    optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=learning_rate)
    scheduler_mlp = StepLR(optimizer_mlp, step_size=step_size_scheduler, gamma=gamma)
else:
    optimizer_mlp = torch.optim.LBFGS(model_mlp.parameters(), lr=1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

# --------------------------------------------------------------------------------------
start_train_time_mlp = time.time()
for epoch in range(epochs):
    total_loss = 0
    progress_bar = tqdm(range(0, len(train_params), batch_size), desc=f'Epoch {epoch + 1}/{epochs}')

    for i in progress_bar:
        batch_params = train_params[i:i + batch_size]
        batch_disp = train_true_ux[i:i + batch_size]
        def closure_mlp():
            optimizer_mlp.zero_grad()
            outputs = model_mlp(batch_params, coordinates)
            loss = criterion3(outputs, batch_disp)
            loss.backward()
            return loss
        loss = optimizer_mlp.step(closure_mlp)
        total_loss += loss.item()
        progress_bar.set_postfix({'Batch Loss': loss.item()})

    avg_loss_mlp = total_loss / (len(train_params) // batch_size)
    loss_vec_mlp.append(avg_loss_mlp)
    tqdm.write(f'End of Epoch {epoch + 1}, Average Loss: {avg_loss_mlp}')
    
    test_loss_mlp = test_model(model_mlp, criterion3, test_params, test_true_ux, coordinates)
    test_loss_vec_mlp.append(test_loss_mlp)
    tqdm.write(f'Test Loss: {test_loss_mlp}')
    
    if opt == "adam":
        scheduler_mlp.step()

    # if (epoch + 1) % 1005 == 0:
    #     print("Heyyyy")
    #     with torch.no_grad():
    #         test_pred_ux_mlp    = model_mlp(test_params, coordinates).cpu().numpy()
    #         scipy.io.savemat('Results_Orthotropic_MLP_' + str(epoch + 1) + '.mat', {'test_pred_ux_mlp': test_pred_ux_mlp, 'loss_vec_mlp':loss_vec_mlp})

end_training_time_mlp = time.time() - start_train_time_mlp

########################################################################################
with torch.no_grad():
    coordinates_cpu     = coordinates.cpu().numpy()
    test_params_cpu     = test_params.cpu().numpy()
    test_pred_ux_rbf    = model_rbf(test_params, coordinates).cpu().numpy()
    test_pred_ux_mlp    = model_mlp(test_params, coordinates).cpu().numpy()
    test_true_ux        = test_true_ux.cpu().numpy() 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Number of learnable parameters in RBF Network: {count_parameters(model_rbf)}')
print(f'Number of learnable parameters in MLP Network: {count_parameters(model_mlp)}')

########################################################################################
scipy.io.savemat('Results_Orthotropic_ModelX.mat', 
                {'coordinates':coordinates_cpu, 
                'test_params':test_params_cpu, 
                'pred_ux_rbf':test_pred_ux_rbf, 
                'pred_ux_mlp':test_pred_ux_mlp,                   
                'test_true_ux':test_true_ux,
                'loss_vec_rbf':loss_vec_rbf,
                'loss_vec_mlp':loss_vec_mlp,
                'test_loss_vec_rbf':test_loss_vec_rbf,
                'test_loss_vec_mlp':test_loss_vec_mlp,
                'end_training_time_rbf':end_training_time_rbf, 
                'end_training_time_mlp':end_training_time_mlp,
                'parameters_rbf':count_parameters(model_rbf),
                'parameters_mlp':count_parameters(model_mlp)})











