"""Wrapper class for pre-tokenized datasets"""
import numpy as np
import torch
from torch.utils.data import Dataset


class MyTokenizedDataset(Dataset):
    def __init__(self, df):
        self.data = {k: torch.LongTensor(np.stack(v.tolist())) for k, v in df.iteritems()}

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

    def __len__(self):
        return self.data['label'].shape[0]
