import torch
import glob
from torchvision.transforms.functional import equalize
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision


class PokemonDataset(Dataset):
    def __init__(self, imgs_paths: str):
        super(PokemonDataset, self).__init__()

        images = []
        for img_path in imgs_paths:
            images.append(torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.GRAY))
        self.inputs = torch.cat(images).float() / 255
        self.inputs = self.inputs.unsqueeze(dim=1)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx, ...]

