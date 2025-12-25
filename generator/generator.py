import time
import torch
import numpy as np
from torch.utils.data import Dataset



class Generator(Dataset):
    def __init__(self, load_data=None, seed=None, n_samples=None, layout=None, inst_type=None, device=None):
        if load_data is None:            
            self.n_samples = n_samples
            self.data = self.generate_data_vectorized(seed, layout, inst_type, device)
        else:
            self.data = torch.load(load_data)
            self.n_samples = self.data.shape[0]


    def generate_data_vectorized(self, seed, layout, inst_type, device):
        n_containers, n_bays, n_rows, n_tiers = layout
        n_stacks = n_bays * n_rows
        assert n_containers <= n_bays * n_rows * n_tiers
        if seed is not None:
            torch.manual_seed(seed)

        clock = time.time()

        data = torch.zeros((self.n_samples, n_stacks, n_tiers), dtype=torch.float32, device=device)

        container_sequences = torch.rand((self.n_samples, n_containers), device=device).argsort(dim=-1).float() + 1

        stack_fill_counts = torch.zeros((self.n_samples, n_stacks), dtype=torch.int32, device=device)

        for j in range(n_containers):
            valid_stacks = stack_fill_counts < n_tiers
            valid_stacks_float = valid_stacks.float()
            
            stack_probs = valid_stacks_float / valid_stacks_float.sum(dim=-1, keepdim=True)
            selected_stacks = torch.multinomial(stack_probs, 1).squeeze(dim=-1)

            tier_positions = stack_fill_counts[torch.arange(self.n_samples, device=device), selected_stacks]
            data[torch.arange(self.n_samples, device=device), selected_stacks, tier_positions] = container_sequences[:, j]
            stack_fill_counts[torch.arange(self.n_samples, device=device), selected_stacks] += 1

        if inst_type == 'upsidedown':
            mask = data > 0
            sorted_data, _ = torch.sort(torch.where(mask, data, torch.inf), dim=-1)
            sorted_data[sorted_data == torch.inf] = 0
            data[:] = sorted_data

        batch_size, total_stacks, feature_dim = data.shape
        assert total_stacks == n_bays * n_rows
        data = data.reshape(batch_size, n_bays, n_rows, feature_dim)

        return data



    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data[index]
