import h5py
import numpy as np
import os
from KAN_RBF6 import RadialBasisFunctionNetwork as RBFKAN
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

time.sleep(5)
dtype = torch.float64  # or torch.float32 or torch.float64
def load_and_concatenate_data(filepath):
    with h5py.File(filepath, 'r') as hdf:
        # Gather coordinates directly into a numpy array
        coordinates = np.array(hdf['coordinates'])

        # Retrieve information about the number of datasets and their shapes
        total_datasets = len(hdf['loads'].keys())
        sample_load = hdf['loads']['load_0']
        sample_solution = hdf['solutions']['solution_0']

        # Determine the shape and datatype for the memmap arrays
        loads_shape = (total_datasets, *sample_load.shape)
        solutions_shape = (total_datasets, *sample_solution.shape)

        # If memmap files exist, delete them first to avoid shape and dtype conflicts
        if os.path.exists('loads20.dat'):
            os.remove('loads20.dat')
        if os.path.exists('solutions20.dat'):
            os.remove('solutions20.dat')

        # Create memmap arrays for loads and solutions
        loads = np.memmap('loads20.dat', dtype=sample_load.dtype, mode='w+', shape=loads_shape)
        solutions = np.memmap('solutions20.dat', dtype=sample_solution.dtype, mode='w+', shape=solutions_shape)

        # Populate memmap arrays directly from HDF5 without loading entire arrays into memory
        for idx in range(total_datasets):
            loads[idx] = hdf['loads'][f'load_{idx}']
            solutions[idx] = hdf['solutions'][f'solution_{idx}']

    return loads, solutions, coordinates

# Example usage:
filepath = '/scratch/da3205/TransientPoisson_KAN/data_90percent.hdf5'
loads, solutions, coordinates = load_and_concatenate_data(filepath)
solutions = solutions.transpose(0,2,1)

# Convert to PyTorch tensors
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
loads = torch.tensor(loads, device=device)[:,:,1]
solutions = torch.tensor(solutions, device=device)
coordinates = torch.tensor(coordinates, device=device)

print(f'loads shape = {loads.shape}')
print(f'solutions shape = {solutions.shape}')
print(f'coordinates shape = {coordinates.shape}')

def split_data(loads, solutions, test_ratio=0.2):
    """ Splits the data into training and testing sets. """
    num_data = loads.shape[0]
    indices = torch.randperm(num_data)
    split = int(num_data * (1 - test_ratio))

    train_indices = indices[:split]
    test_indices = indices[split:]

    train_loads = loads[train_indices]
    train_solutions = solutions[train_indices]
    test_loads = loads[test_indices]
    test_solutions = solutions[test_indices]

    return train_loads, train_solutions, test_loads, test_solutions

print(f'loads max = {loads.max()}, loads min = {loads.min()}')
print(f'coordinates max = {coordinates.max()}, coordinates min = {coordinates.min()}')

train_loads, train_solutions, test_loads, test_solutions = split_data(loads, solutions, test_ratio=0.2)
print(f'train_loads shape = {train_loads.shape}, train_solutions shape = {train_solutions.shape}')
print(f'test_loads shape = {test_loads.shape}, test_solutions shape = {test_solutions.shape}')

def define_kan_width(input_dim, W, repeat_hid, output_dim):
    width = [input_dim] + [W] * repeat_hid + [output_dim]
    return width

input_dim_trunk  = coordinates.shape[1]
input_dim_branch = loads.shape[1]
HD = 4
hid_trunk = 5
num_layer_trunk  = 1
hid_branch = 5
num_layer_branch = 1
trunk_min_grid = coordinates.min()
trunk_max_grid = coordinates.max()
branch_min_grid = loads.min()
branch_max_grid = loads.max()

width_trunk = define_kan_width(input_dim_trunk, hid_trunk, num_layer_trunk, HD)
width_branch = define_kan_width(input_dim_branch, hid_branch, num_layer_branch, input_dim_branch*HD)

print(f'hid dim = {hid_trunk}, num_layer = {num_layer_trunk}')
print(f'width_trunk = {width_trunk}, width_branch = {width_branch}')
grid_opt = False 
grid_count = 5
init_scale = 0.01
print(f'init_scale = {init_scale}')

# Hyperparameters
learning_rate = 1e-3
batch_size    = 64
epochs        = 10_000
gamma         = 0.5
step_size     = 1_000
seed          = 42

train_loads = train_loads.clone().detach().requires_grad_(True).to(dtype)
train_solutions = train_solutions.clone().detach().requires_grad_(True).to(dtype)
test_loads = test_loads.clone().detach().requires_grad_(True).to(dtype)
test_solutions = test_solutions.clone().detach().requires_grad_(True).to(dtype)

class DeepONet(nn.Module):
    def __init__(self):
        super(DeepONet, self).__init__()
        self.trunk_net  = RBFKAN(hidden_layers=width_trunk,  dtype=dtype, apply_base_update=False, init_scale=init_scale,
                                  min_grid=trunk_min_grid, max_grid=trunk_max_grid, grid_count=grid_count, grid_opt=grid_opt).to(device)
        self.branch_net = RBFKAN(hidden_layers=width_branch, dtype=dtype, apply_base_update=False, init_scale=init_scale,
                                  min_grid=branch_min_grid,  max_grid=branch_max_grid,  grid_count=grid_count, grid_opt=grid_opt).to(device)

    def forward(self, x_branch, x_trunk):
        y_branch = self.branch_net(x_branch)
        y_branch = y_branch.view(-1, train_loads.shape[1], HD)
        y_trunk = self.trunk_net(x_trunk)
        
        if y_branch.shape[-1] != y_trunk.shape[-1]:
            raise AssertionError("Output sizes of branch net and trunk net do not match.")

        # Einsum operation to combine outputs into the desired shape
        Y = torch.einsum('bik,nk->bni', y_branch, y_trunk)

        return Y

model = DeepONet()

class L2NormLoss(nn.Module):
    def __init__(self):
        super(L2NormLoss, self).__init__()

    def forward(self, output, target):
        norm = torch.norm(output - target, p=2)
        num_elements = target.shape[0]  # This gives the total number of elements in the output tensor
        return norm / num_elements

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

criterion1 = nn.MSELoss()
criterion2 = L2NormLoss()
criterion3 = RMSDLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

def test_model(model, criterion, test_loads, test_solutions, coordinates, batch_size):
    """Evaluates the model on the test dataset in batches."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    num_batches = len(test_loads) // batch_size + 1
    with torch.no_grad():  # No need to track gradients for testing
        for i in tqdm(range(0, len(test_loads), batch_size), desc='Testing'):
            batch_loads = test_loads[i:i + batch_size]
            batch_solutions = test_solutions[i:i + batch_size]
            outputs = model(batch_loads, coordinates)
            loss = criterion(outputs, batch_solutions)
            total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return avg_loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(seed)

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
num_learnable_parameters = count_learnable_parameters(model)
print(f'The number of learnable parameters in the model: {num_learnable_parameters}')

# Assuming your model, optimizer, loss function, scheduler, and data are defined and set up
train_losses = []
test_losses = []
mean_test_loss = 0
start_time = time.time()
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    progress_bar = tqdm(range(0, len(train_loads), batch_size), desc=f'Epoch {epoch + 1}/{epochs}')
    for i in progress_bar:
        batch_loads = train_loads[i:i + batch_size]
        batch_solutions = train_solutions[i:i + batch_size]

        # Define the closure for the L-BFGS optimizer
        def closure():
            optimizer.zero_grad()
            outputs = model(batch_loads, coordinates)
            loss = criterion3(outputs, batch_solutions)
            loss.backward()
            return loss

        # Perform an optimization step
        loss = optimizer.step(closure)
        total_loss += loss.item()

        # Update the progress bar with the current batch loss
        progress_bar.set_postfix({'Batch Loss': loss.item()})

    # Calculate the average loss for the epoch
    avg_loss = total_loss / (len(train_loads) // batch_size + 1)
    train_losses.append(avg_loss)
    tqdm.write(f'End of Epoch {epoch + 1}, Average Loss: {avg_loss}')

    # Evaluate the model on the testing dataset
    test_loss = test_model(model, criterion3, test_loads, test_solutions, coordinates, batch_size)
    test_losses.append(test_loss)
    tqdm.write(f'Test Loss: {test_loss}')
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
# Create a DataFrame from the losses
loss_df = pd.DataFrame({
    'Epoch': range(1, epochs + 1),
    'Train Loss': train_losses,
    'Test Loss': test_losses
})
# Save the DataFrame to a CSV file
loss_df.to_csv('DeepOKAN_training_testing_losses_seed42.csv', index=False)
print('Training and testing losses have been saved to DeepOKAN_training_testing_losses.csv')
      
# Define a function to get predictions in batches and save directly to file
def get_predictions_and_save(model, test_loads, coordinates, batch_size, filename_pattern):
    model.eval()  # Set the model to evaluation mode

    # Number of splits
    num_splits = 4
    split_size = len(test_loads) // num_splits
    
    # Prepare HDF5 files for writing
    for i in range(num_splits):
        filename = filename_pattern.format(i + 1)
        with h5py.File(filename, 'w') as h5f:
            h5f.create_dataset('coordinates', data=coordinates.cpu().numpy(), compression='gzip')
            h5f.create_dataset('test_loads', shape=(split_size, *test_loads.shape[1:]), dtype=np.float32, compression='gzip')
            h5f.create_dataset('test_pred', shape=(split_size, *model(test_loads[:batch_size], coordinates).shape[1:]), dtype=np.float32, compression='gzip')
            h5f.create_dataset('test_solutions', shape=(split_size, *test_solutions.shape[1:]), dtype=np.float32, compression='gzip')

    # Process and save predictions in batches
    batch_counter = 0
    with torch.no_grad():  # No need to track gradients for prediction
        for i in tqdm(range(0, len(test_loads), batch_size), desc='Predicting'):
            batch_loads = test_loads[i:i + batch_size]
            batch_solutions = test_solutions[i:i + batch_size]
            batch_predictions = model(batch_loads, coordinates).cpu().numpy()

            # Determine the corresponding HDF5 file and dataset indices
            file_idx = batch_counter // split_size
            dataset_idx = batch_counter % split_size
            batch_counter += batch_loads.shape[0]

            filename = filename_pattern.format(file_idx + 1)
            with h5py.File(filename, 'a') as h5f:
                # Determine the actual size to write to avoid broadcasting issues
                actual_size = min(split_size - dataset_idx, batch_loads.shape[0])
                h5f['test_loads'][dataset_idx:dataset_idx + actual_size] = batch_loads.cpu().numpy()[:actual_size]
                h5f['test_pred'][dataset_idx:dataset_idx + actual_size] = batch_predictions[:actual_size]
                h5f['test_solutions'][dataset_idx:dataset_idx + actual_size] = batch_solutions.cpu().numpy()[:actual_size]
    
    print("Predictions saved successfully.")

# Assume training is complete
time.sleep(5)
print('Training is complete')

# Define the filename pattern for saving splits
filename_pattern = 'transient_Poisson_results_rbfkan_file_seed42_part_{}.h5'

# Transfer test data to CPU and obtain predictions in batches, saving directly to files
get_predictions_and_save(model, test_loads, coordinates, batch_size, filename_pattern)

# Print confirmation
time.sleep(5)
print("Numpy arrays are ready and saved to files.")

# Load and print shapes for verification
for i in range(1, 5):
    filename = filename_pattern.format(i)
    with h5py.File(filename, 'r') as h5f:
        coordinates_cpu = np.array(h5f['coordinates'])
        test_loads_cpu = np.array(h5f['test_loads'])
        test_pred_cpu = np.array(h5f['test_pred'])
        test_solutions_cpu = np.array(h5f['test_solutions'])
        print(f'File {filename}:')
        print(f'coordinates_cpu shape = {coordinates_cpu.shape}')
        print(f'test_loads_cpu shape = {test_loads_cpu.shape}')
        print(f'test_pred_cpu shape = {test_pred_cpu.shape}')
        print(f'test_solutions_cpu shape = {test_solutions_cpu.shape}')