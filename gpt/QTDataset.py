import torch
from torch.utils.data import Dataset


class QTDataset(Dataset):
    def __init__(self, data, truncate=False):
        super(QTDataset, self).__init__()
        self.inputs = data[:, 0, :]
        self.masks = data[:, 1, :]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.masks[idx]
