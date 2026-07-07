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
        self.a_values_np = self.a_values.detach().cpu().numpy() if torch.is_tensor(self.a_values) else np.asarray(self.a_values)
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
            return torch.from_numpy(np.load(self.file_paths[idx]).astype(np.float32))
        return self.data_tensor[idx]

    def sample_pair_indices(self, rng=None):
        rng = rng or np.random
        i = rng.randint(0, self.N - 1)
        max_possible_step = min(self.max_step, self.N - 1 - i)
        if max_possible_step <= 1:
            return i, i + 1

        max_j = i + max_possible_step
        min_delta_a = self.a_values_np[i + 1] - self.a_values_np[i]
        max_delta_a = self.a_values_np[max_j] - self.a_values_np[i]
        if min_delta_a <= 0 or max_delta_a <= min_delta_a:
            j = rng.randint(i + 1, max_j + 1)
        else:
            log_delta_a = rng.uniform(np.log(min_delta_a), np.log(max_delta_a))
            target_a = self.a_values_np[i] + np.exp(log_delta_a)
            j = np.searchsorted(self.a_values_np, target_a, side="left")
            j = max(i + 1, min(j, max_j))
        return i, j

    def make_pair(self, i, j):
        rho_in = self._load_snapshot(i)
        rho_out = self._load_snapshot(j)
        delta_a = self.a_values[j] - self.a_values[i]

        delta_a_channel = torch.full_like(rho_in, delta_a)
        x = torch.stack([rho_in, delta_a_channel], dim=0)
        y = rho_out.unsqueeze(0)

        return x, y

    def __getitem__(self, idx):
        if self.is_train:
            i, j = self.sample_pair_indices()
        else:
            i = idx
            j = min(i + 1, self.N - 1)
            if j <= i:
                j = min(i + 1, self.N - 1)

        return self.make_pair(i, j)


def load_miguel_data(data_dir, dry_run=False, lazy=False, split_strategy="chronological", train_fraction=0.8):
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
    def select_data(data, indices):
        if torch.is_tensor(data):
            return data[indices]
        return [data[i] for i in indices.tolist()]

    if split_strategy == "interleaved":
        test_stride = max(2, round(1.0 / (1.0 - train_fraction)))
        test_idx = np.arange(num_snapshots) % test_stride == test_stride - 1
        train_idx = ~test_idx
        train_idx = torch.from_numpy(np.flatnonzero(train_idx))
        test_idx = torch.from_numpy(np.flatnonzero(test_idx))
        m_train_data = select_data(m_files, train_idx)
        m_train_as = m_as[train_idx]
        m_test_data = select_data(m_files, test_idx)
        m_test_as = m_as[test_idx]
    elif split_strategy == "chronological":
        train_split = int(num_snapshots * train_fraction)
        m_train_data = m_files[:train_split]
        m_train_as = m_as[:train_split]
        m_test_data = m_files[train_split:]
        m_test_as = m_as[train_split:]
    else:
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")

    print(f"  Split strategy: {split_strategy}")
    print(f"  Train snapshots: {len(m_train_as)} | Test snapshots: {len(m_test_as)}")
    return m_train_data, m_train_as, m_test_data, m_test_as
