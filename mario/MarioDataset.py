import torch
from torch.utils.data import Dataset


class MarioDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.from_numpy(inputs.astype('float32'))
        self.targets = torch.from_numpy(targets.astype('long'))

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx, ...], self.targets[idx, ...]
