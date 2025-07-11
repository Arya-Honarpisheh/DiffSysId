import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import torch

def get_dataloader(dataset, seed=0, nfold=None, batch_size=32):
    """
    Get DataLoader for a dataset.
    Args:
        dataset: The dataset to load.
        nfold (int, optional): Fold number for cross-validation.
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