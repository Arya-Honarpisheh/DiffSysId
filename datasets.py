import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, Dataset, Subset
import torch
from tqdm import tqdm

# =============================================================================
# LOTKA-VOLTERRA MODEL FUNCTIONS
# =============================================================================

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
            # add multiplicative lognormal noise based on the signal to noise ratio
            # noise_std = sqrt(ln(1 + 1/SNR)), noise_mean = -sigma^2/2 to remove bias
            noise_std1 = np.sqrt(np.log(1 + 1 / config['noise_snr'][0]))
            noise_std2 = np.sqrt(np.log(1 + 1 / config['noise_snr'][1]))
            noise_mean1 = -noise_std1**2 / 2
            noise_mean2 = -noise_std2**2 / 2
            x1 += np.random.normal(noise_mean1, noise_std1, size=x1.shape)
            x2 += np.random.normal(noise_mean2, noise_std2, size=x2.shape)
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

# =============================================================================
# LORENZ MODEL FUNCTIONS
# =============================================================================

def LorenzModel(x_init, total_time, dt, parameters):
    """
    Lorenz Attractor model simulation with given parameters.
    Args:
        x_init (list): Initial conditions [x, y, z].
        total_time (float): Total time for simulation.
        dt (float): Time step for simulation.
        parameters (list): Parameters [rho, alpha, beta].
    """

    num_T = int(total_time/dt) # number of time steps

    def model_derivative(t, x, rho, alpha, beta):
        x1, x2, x3 = x
        dxdt = [alpha * (x2 - x1), 
                x1 * (rho - x3) - x2,
                x1 * x2 - beta * x3]
        return dxdt
    
    sol = solve_ivp(
    model_derivative,
    [0, total_time],
    x_init,
    args=tuple(parameters),  # ← correctly unpacks the parameters
    method='RK45',
    t_eval=np.linspace(0, total_time, num_T + 1)
    )

    return sol.t, sol.y[0,:], sol.y[1,:], sol.y[2,:]

class LorenzDataset(Dataset):
    def __init__(self, num_samples, total_time, dt, config):
        """
        Generate synthetic data for Lorenz model.
        Args:
            num_samples (int): Number of samples to generate.
            total_time (float): Total time for simulation.
            dt (float): Time step for simulation.
            parameters (list): Parameters [rho, alpha, beta].
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
            f"Generating Lorenz dataset with {num_samples} samples\n"
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
            # run the Lorenz model
            t, x1, x2, x3 = LorenzModel(x_init, total_time, dt, parameters)
            # randomly choose a subset of the data based on the config['data_sparsity']
            indices = np.random.choice(len(t), config['seq_len'], replace=False)
            indices = np.sort(indices)  # sort indices to maintain order
            t = t[indices]
            x1 = x1[indices]
            x2 = x2[indices]
            x3 = x3[indices]
            # add noise to the data based on the signal to noise ratio
            noise_std1 = np.max(np.abs(x1)) / config['noise_snr'][0]
            noise_std2 = np.max(np.abs(x2)) / config['noise_snr'][1]
            noise_std3 = np.max(np.abs(x3)) / config['noise_snr'][2]
            x1 += np.random.normal(0, noise_std1, size=x1.shape)
            x2 += np.random.normal(0, noise_std2, size=x2.shape)
            x3 += np.random.normal(0, noise_std3, size=x3.shape)
            # append the data to the list as a torch tensor
            x = np.column_stack((x1, x2, x3)) # (L, 3)
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

# =============================================================================
# GENERAL DATASET UTILITIES
# =============================================================================

def get_dataloader(dataset, seed=0, nfold=None, batch_size=32):
    """
    Get DataLoader for a dataset.
    Args:
        dataset: The dataset to load.
        nfold (int, optional): Fold number for 5-fold cross-validation.
        batch_size (int): Batch size for DataLoader.
    """
    # set the seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    indlist = np.arange(len(dataset))

    # 20% test     70% train     10% validation

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))
    # randomly choose train and validation sets
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    train_dataset = Subset(dataset, train_index)
    valid_dataset = Subset(dataset, valid_index)
    test_dataset = Subset(dataset, test_index)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=g)

    return train_loader, valid_loader, test_loader

class NormalizedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        all_x = torch.cat([sample['x'] for sample in dataset], dim=0)  # shape: (total_time_points, ts_dim)
        self.x_mean = all_x.mean(dim=0)   # shape: (ts_dim)
        self.x_std = all_x.std(dim=0)     # shape: (ts_dim)

        all_parameters = torch.stack([sample['parameters'] for sample in dataset], dim=0)
        self.parameters_mean = all_parameters.mean(dim=0)
        self.parameters_std = all_parameters.std(dim=0)

        all_x_init = torch.stack([sample['x_init'] for sample in dataset], dim=0)
        self.x_init_mean = all_x_init.mean(dim=0)
        self.x_init_std = all_x_init.std(dim=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x = (sample['x'] - self.x_mean) / self.x_std
        x_init = (sample['x_init'] - self.x_init_mean) / self.x_init_std
        parameters = (sample['parameters'] - self.parameters_mean) / self.parameters_std
        return {
            'parameters': parameters, # (B, P)
            'time': sample['time'], # (B, L)
            'x_init': x_init, # (B, ts_dim)
            'x': x # (B, L, ts_dim)
        }

    def get_param_stats(self):
        return self.parameters_mean, self.parameters_std
