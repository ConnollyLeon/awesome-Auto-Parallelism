import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np


class RandomDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1024

    def __getitem__(self, idx):
        pt_data = dict()
        pt_data['input_ids'] = torch.tensor(np.random.randint(1, 30000, (1, 512)), dtype=torch.long)
        pt_data['token_type_ids'] = torch.tensor(np.random.randint(0, 1, (1, 512)), dtype=torch.long)
        pt_data['attention_mask'] = torch.tensor(np.random.random((1, 512)))
        label = torch.tensor(np.random.randint(0, 1, (1, 1)))
        masked_lm_ids = torch.tensor(np.random.randint(0, 30000, (1, 80)))
        masked_lm_positions = torch.tensor(np.random.randint(0, 511, (1, 80)))
        masked_lm_weights = torch.tensor(np.random.random((1, 80)))

        return pt_data, label, masked_lm_ids, masked_lm_positions, masked_lm_weights
