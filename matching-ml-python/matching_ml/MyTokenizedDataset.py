import numpy as np
import torch
from torch.utils.data import Dataset


class MyTokenizedDataset(Dataset):
    def __init__(self, df):
        self.data = {k: torch.LongTensor(np.stack(v.tolist())) for k, v in df.iteritems()}
        # self.position_ids = torch.IntTensor(np.stack(df['position_ids'].tolist()))
        # self.input_ids = torch.IntTensor(np.stack(df['input_ids'].tolist()))
        # self.attention_mask = torch.IntTensor(np.stack(df['attention_mask'].tolist()))
        # self.labels = torch.LongTensor(df['label'].values)

    def __getitem__(self, idx):
        # return {
        #     'position_ids': self.position_ids[idx],
        #     'input_ids': self.input_ids[idx],
        #     'attention_mask': self.attention_mask[idx],
        #     'label': self.labels[idx],
        # }
        return {k: v[idx] for k, v in self.data.items()}

    def __len__(self):
        return self.data['label'].shape[0]
