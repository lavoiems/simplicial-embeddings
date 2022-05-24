import torch
from torch import nn
import tensorflow as tf
import numpy as np


@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module) -> Tuple[torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """
    mean = [255*0.485, 255*0.456, 255*0.406]
    std = [255*0.229, 255*0.224, 255*0.225]

    model.eval()
    backbone_features, proj_features, labels = [], [], []
    emb_features = []
    for batch in tqdm(loader):
        im, lab = batch['image'], batch['label']
        im = im[0]
        h, w = tf.shape(im)[0], tf.shape(im)[1]
        ratio = (
            tf.cast(256, tf.float32) /
            tf.cast(tf.minimum(h, w), tf.float32))
        h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
        w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
        im = tf.image.resize(im[None], (h,w), method='bicubic')
        im = im[0]

        h, w = 224, 224
        dy = (tf.shape(im)[0] - h) // 2
        dx = (tf.shape(im)[1] - w) // 2
        im = tf.image.crop_to_bounding_box(im, dy, dx, h, w)
        im = im[None]

        im = (im - mean) / std
        im = np.transpose(im, [0, 3, 1, 2]).astype(np.float32).copy()
        im = torch.from_numpy(im)
        lab = torch.from_numpy(lab.numpy())
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        outs = model(im)
        backbone_features.append(outs["feats"].detach())
        labels.append(lab)
        proj_features.append(outs["z"])
        emb = outs.get('emb', None)
        if emb is not None:
            emb_features.append(emb.half())
    model.train()
    backbone_features = torch.cat(backbone_features).cpu().numpy()
    if len(emb_features) > 0:
        emb_features = torch.cat(emb_features).half()
    labels = torch.cat(labels).cpu().numpy()
    return backbone_features, proj_features, emb_features, labels


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.features = data['features']
        self.targets = data['targets']

    def __len__():
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def prepare_data(loader, model):
    _, _, features_emb, targets = extract_feature(loader, model)
    data = {"features": train_features_emb, "targets": train_targets}
    dataset = FeatureDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=4)
    return loader

