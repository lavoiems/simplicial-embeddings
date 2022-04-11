import torch
from torch.utils.data.dataset import Dataset


class ArrayDataset(Dataset):
    def __init__(self, array, transform=None):
        images = array['imgs']
        labels = array['latents_classes']
        if images.ndim == 3:
            images = torch.from_numpy(images)
            images = torch.concatenate([images, images, images], 1)
        else:
            images = torch.from_numpy(images.transpose((0, 3, 1, 2))).contiguous()
        self.images = images.to(dtype=torch.get_default_dtype())
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

