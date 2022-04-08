import numpy as np
from torch.utils.data.dataset import Dataset


class CompositionalDataset(Dataset):
    def __init__(self, array, transform=None):
        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
