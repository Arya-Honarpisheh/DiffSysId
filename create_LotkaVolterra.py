
from dataset_LotkaVolterra import LotkaVolterraDataset
import torch
import os
import yaml

num_samples = 10000 # Number of samples to generate
total_time = 20.0
dt = 0.01
seed = 0
config_dataset = {'seed': seed,
          'x_init_low': [5, 5], 'x_init_high': [8, 8],
          'parameters_low': [0.5, 0.1, 0.1, 0.5], 
          'parameters_high': [2.0, 0.5, 0.5, 1.0],
          'data_sparsity': 0.03, 'noise_snr': [999999, 999999]}

# Generate Data
dataset = LotkaVolterraDataset(num_samples, total_time, dt, config_dataset)

data_folder = './data/LotkaVolterra'+ '_S_' + str(num_samples) + '_T_' + str(total_time) + '_dt_' + str(dt) + '_seed_' + str(seed)
os.makedirs(data_folder, exist_ok=True)
torch.save(dataset, os.path.join(data_folder, 'dataset.pt'))
with open(os.path.join(data_folder, 'config'), "w") as f:
    yaml.dump(config_dataset, f, indent=4)

