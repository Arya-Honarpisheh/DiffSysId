
from dataset_Lorenz import LorenzDataset
import torch
import os
import yaml
import numpy as np

num_samples = 10000 # Number of samples to generate
total_time = 20.0
dt = 0.001
seq_len = 100
seed = 0
config_dataset = {'seed': seed,
          'x_init_low': [-15, -10, 20], 'x_init_high': [-5, -1, 50],
          'parameters_low': [0, 5, 1], 
          'parameters_high': [60, 25, 8],
          'seq_len': seq_len, 'noise_snr': [15, 15, 15]}

# Generate Data
dataset = LorenzDataset(num_samples, total_time, dt, config_dataset)

data_folder = './data/Lorenz'+ '_S_' + str(num_samples) + '_T_' + str(total_time) + '_dt_' + str(dt) + '_seq_len_'+ str(seq_len) + '_seed_' + str(seed)
os.makedirs(data_folder, exist_ok=True)
torch.save(dataset, os.path.join(data_folder, 'dataset.pt'))
with open(os.path.join(data_folder, 'config.yaml'), "w") as f:
    yaml.dump(config_dataset, f, indent=4)

print(f"Dataset created at: {data_folder}/dataset.pt")
print(f"Configuration saved at: {data_folder}/config")

