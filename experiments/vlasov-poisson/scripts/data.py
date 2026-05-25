import numpy as np
import torch
from torch.utils.data import Dataset


class VlasovPoissonDataset(Dataset):
    def __init__(self, data_tensor, a_values, is_train=True, max_step=100, num_samples=10000, lazy=False):
        self.lazy = lazy
        if lazy:
            self.file_paths = data_tensor
            self.a_values = a_values
        else:
            self.data_tensor = data_tensor
            self.a_values = a_values
        self.is_train = is_train
        self.max_step = max_step
        self.num_samples = num_samples
        self.N = len(a_values)

    def __len__(self):
        if self.is_train:
            return self.num_samples
        else:
            return max(1, self.N - 20)

    def _load_snapshot(self, idx):
        if self.lazy:
            return np.load(self.file_paths[idx]).astype(np.float32)
        return self.data_tensor[idx]

    def __getitem__(self, idx):
        if self.is_train:
            i = np.random.randint(0, self.N - 1)
            j = np.random.randint(i + 1, min(i + self.max_step + 1, self.N))
        else:
            i = idx
            j = min(i + max(1, self.max_step // 2), self.N - 1)
            if j <= i:
                j = min(i + 1, self.N - 1)

        rho_in = torch.from_numpy(self._load_snapshot(i))
        rho_out = torch.from_numpy(self._load_snapshot(j))
        delta_a = self.a_values[j] - self.a_values[i]

        delta_a_channel = torch.full_like(rho_in, delta_a)
        x = torch.stack([rho_in, delta_a_channel], dim=0)
        y = rho_out.unsqueeze(0)

        return x, y


def load_miguel_data(data_dir, dry_run=False, lazy=False):
    m_dir = data_dir / "miguel_64" / "density"
    m_files = sorted(list(m_dir.glob("*.npy")), key=lambda f: float(f.stem.split("_")[1]), reverse=True)
    m_zs = [float(f.stem.split("_")[1]) for f in m_files]
    m_as = torch.tensor([1.0 / (1.0 + z) for z in m_zs], dtype=torch.float32)

    if dry_run:
        m_files = m_files[:100]
        m_as = m_as[:100]

    if lazy:
        m_files = [f for f in m_files]
        print(f"  Lazy mode: {len(m_files)} file paths registered (not loaded)")
    else:
        snapshots = []
        import time
        t0 = time.time()
        for f in m_files:
            snapshots.append(torch.from_numpy(np.load(f).astype(np.float32)))
        m_files = torch.stack(snapshots, dim=0)
        print(f"  Eager mode: loaded {len(m_files)} snapshots in {time.time() - t0:.2f}s. Shape: {m_files.shape}")

    num_snapshots = len(m_as)
    train_split = int(num_snapshots * 0.8)

    m_train_data = m_files[:train_split]
    m_train_as = m_as[:train_split]
    m_test_data = m_files[train_split:]
    m_test_as = m_as[train_split:]

    print(f"  Train snapshots: {len(m_train_as)} | Test snapshots: {len(m_test_as)}")
    return m_train_data, m_train_as, m_test_data, m_test_as