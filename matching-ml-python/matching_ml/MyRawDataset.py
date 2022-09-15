from torch.utils.data import Dataset


class MyRawDataset(Dataset):
    def __init__(self, texts_left, texts_right, labels=None):
        self.texts_left = texts_left
        self.texts_right = texts_right
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'text_left': self.texts_left[idx],
            'text_right': self.texts_right[idx],
        }
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.texts_left)
