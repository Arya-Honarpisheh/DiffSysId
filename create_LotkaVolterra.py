
from datasets import LotkaVolterraDataset
import torch
import os
import yaml

num_samples = 10000 # Number of samples to generate
total_time = 20.0
dt = 0.001
seq_len = 60
seed = 0
config_dataset = {'seed': seed,
          'x_init_low': [20, 5], 'x_init_high': [100, 50],
          'parameters_low': [0.1, 0.01, 0.01, 0.1], 
          'parameters_high': [3.0, 0.5, 0.5, 3.0],
          'seq_len': seq_len, 'noise_snr': [20, 20]}

# Generate Data
dataset = LotkaVolterraDataset(num_samples, total_time, dt, config_dataset)

data_folder = './data/LotkaVolterra'+ '_S_' + str(num_samples) + '_T_' + str(total_time) + '_dt_' + str(dt) + '_seq_len_'+ str(seq_len) + '_seed_' + str(seed)
os.makedirs(data_folder, exist_ok=True)
torch.save(dataset, os.path.join(data_folder, 'dataset.pt'))
with open(os.path.join(data_folder, 'config.yaml'), "w") as f:
    yaml.dump(config_dataset, f, indent=4)

print(f"Dataset created at: {data_folder}/dataset.pt")
print(f"Configuration saved at: {data_folder}/config")

