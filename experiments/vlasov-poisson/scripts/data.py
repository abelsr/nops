import numpy as np
import torch
from torch.utils.data import Dataset


class VlasovPoissonDataset(Dataset):
    def __init__(self, data_tensor, a_values, is_train=True, max_step=100, num_samples=10000):
        self.data_tensor = data_tensor
        self.a_values = a_values
        self.is_train = is_train
        self.max_step = max_step
        self.num_samples = num_samples
        self.N = data_tensor.shape[0]

    def __len__(self):
        if self.is_train:
            return self.num_samples
        else:
            return max(1, self.N - 20)

    def __getitem__(self, idx):
        if self.is_train:
            i = np.random.randint(0, self.N - 1)
            j = np.random.randint(i + 1, min(i + self.max_step + 1, self.N))
        else:
            i = idx
            j = min(i + max(1, self.max_step // 2), self.N - 1)
            if j <= i:
                j = min(i + 1, self.N - 1)

        rho_in = self.data_tensor[i]
        rho_out = self.data_tensor[j]
        delta_a = self.a_values[j] - self.a_values[i]

        delta_a_channel = torch.full_like(rho_in, delta_a)
        x = torch.stack([rho_in, delta_a_channel], dim=0)
        y = rho_out.unsqueeze(0)

        return x, y