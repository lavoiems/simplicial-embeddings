import numpy as np
from torch.utils.data.dataset import Dataset


class CompositionalDataset(Dataset):
    def __init__(self, array, transform=None):
        images = array['imgs']
        labels = array['latents_classes']
        self.images = np.asarray(images)
        self.labels = np.asarray(labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
