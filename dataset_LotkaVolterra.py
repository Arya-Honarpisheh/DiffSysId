import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, Dataset, Subset
import torch
from tqdm import tqdm

def LVModel(x_init, total_time, dt, parameters):
    """
    Lotka-Volterra model simulation with given parameters.
    Args:
        x_init (list): Initial conditions [prey, predator].
        total_time (float): Total time for simulation.
        dt (float): Time step for simulation.
        parameters (list): Parameters [alpha, beta, delta, gamma].
    """

    num_T = int(total_time/dt) # number of time steps

    def model_derivative(t, x, alpha, beta, delta, gamma):
        x1, x2 = x
        dxdt = [alpha * x1 - beta * x1 * x2, 
                delta * x1 * x2 - gamma * x2]
        return dxdt
    
    sol = solve_ivp(
    model_derivative,
    [0, total_time],
    x_init,
    args=tuple(parameters),  # ← correctly unpacks [a, b, c, d] to 4 args
    method='RK45',
    t_eval=np.linspace(0, total_time, num_T + 1)
    )

    return sol.t, sol.y[0,:], sol.y[1,:]

class LotkaVolterraDataset(Dataset):
    def __init__(self, num_samples, total_time, dt, config):
        """
        Generate synthetic data for Lotka-Volterra model.
        Args:
            num_samples (int): Number of samples to generate.
            total_time (float): Total time for simulation.
            dt (float): Time step for simulation.
            parameters (list): Parameters [alpha, beta, delta, gamma].
            config (dict): Configuration dictionary.
        """
        # set the seed
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        # initialize lists to store data
        self.time = []
        self.x = []
        self.x_init = []
        self.parameters = []
        # print the information for generating dataset
        print(
            f"Generating Lotka-Volterra dataset with {num_samples} samples\n"
            f"Total time: {total_time}, dt: {dt}, sequence length: {config['seq_len']}\n"
            f"Noise SNR: {config['noise_snr']}\n"
            f"Initial conditions range: [{config['x_init_low']}, {config['x_init_high']}]\n"
            f"Parameters range: [{config['parameters_low']}, {config['parameters_high']}]"
        )
        for _ in tqdm(range(num_samples), desc="Generating dataset"):
            # generate random initial conditions
            x_init = np.random.uniform(config['x_init_low'], config['x_init_high'])
            # genrate random parameters
            parameters = np.random.uniform(config['parameters_low'], config['parameters_high'])
            # run the Lotka-Volterra model
            t, x1, x2 = LVModel(x_init, total_time, dt, parameters)
            # randomly choose a subset of the data based on the config['data_sparsity']
            indices = np.random.choice(len(t), config['seq_len'], replace=False)
            indices = np.sort(indices)  # sort indices to maintain order
            t = t[indices]
            x1 = x1[indices]
            x2 = x2[indices]
            # avoid log(0)
            x1 = np.clip(x1, a_min=1e-6, a_max=None)
            x2 = np.clip(x2, a_min=1e-6, a_max=None)
            # apply logarithm to the data
            x1 = np.log(x1)
            x2 = np.log(x2)
            # add noise to the data based on the signal to noise ratio
            noise_std1 = - np.log(1 - 1 / config['noise_snr'][0])
            noise_std2 = - np.log(1 - 1 / config['noise_snr'][1])
            x1 += np.random.normal(0, noise_std1, size=x1.shape)
            x2 += np.random.normal(0, noise_std2, size=x2.shape)
            # turn back the data to linear scale
            x1 = np.exp(x1)
            x2 = np.exp(x2)
            # append the data to the list as a torch tensor
            x = np.column_stack((x1, x2)) # (L, 2)
            self.time.append(torch.tensor(t, dtype=torch.float32))
            self.x.append(torch.tensor(x, dtype=torch.float32))
            self.x_init.append(torch.tensor(x_init, dtype=torch.float32))
            self.parameters.append(torch.tensor(parameters, dtype=torch.float32))

    def __len__(self):
        return len(self.time) 

    def __getitem__(self, idx):
        return {'parameters': self.parameters[idx],
                'time': self.time[idx],
                'x_init': self.x_init[idx],
                'x': self.x[idx]}