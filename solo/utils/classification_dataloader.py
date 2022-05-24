# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder


def build_custom_pipeline():
    """Builds augmentation pipelines for custom data.
    If you want to do exoteric augmentations, you can just re-write this function.
    Needs to return a dict with the same structure.
    """

    pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
    }
    return pipeline


def prepare_transforms(dataset: str, pretrain_augs: bool) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    if pretrain_augs:
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)

        train_transform = transforms.Compose([
           transforms.RandomResizedCrop(
               (32, 32),
               scale=(0.08, 1.0),
               interpolation=transforms.InterpolationMode.BICUBIC,
           ),
           transforms.RandomApply(
               [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
               p=0.8,
           ),
           transforms.RandomGrayscale(p=0.2),
           transforms.RandomHorizontalFlip(p=0.5),
           transforms.ToTensor(),
           transforms.Normalize(mean, std),
        ])

    else:
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32,
                                             scale=(0.08, 1.0),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    cifar_pipeline = {
        "T_train": train_transform,
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
    }

    custom_pipeline = build_custom_pipeline()

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "stl10": stl_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "custom": custom_pipeline,
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    print('transform training', T_train)

    return T_train, T_val


def prepare_linear(
    dataset: str,
    batch_size: int,
    validation: bool,
):
    if validation:
        train_split = 'train[:90%]'
        val_split = 'train[90%:]'
    else:
        test_splits = {
            'food101': 'validation',
        }
        train_split = 'train'
        val_split = test_splits.get(dataset, 'test')
    tf_ds_train = tfds.load(dataset, batch_size=batch_size, split=train_split)
    tf_ds_val = tfds.load(dataset, batch_size=batch_size, split=val_split)
    return tf_ds_train, tf_ds_val


def prepare_datasets(
    dataset: str,
    T_train: Callable,
    T_val: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    download: bool = True,
    validation: bool = False,
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        data_dir Optional[Union[str, Path]]: path where to download/locate the dataset.
        train_dir Optional[Union[str, Path]]: subpath where the training data is located.
        val_dir Optional[Union[str, Path]]: subpath where the validation data is located.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if data_dir is None:
        sandbox_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        data_dir = sandbox_dir / "datasets"
    else:
        data_dir = Path(data_dir)

    if train_dir is None:
        train_dir = Path(f"{dataset}/train")
    else:
        train_dir = Path(train_dir)

    if val_dir is None:
        val_dir = Path(f"{dataset}/val")
    else:
        val_dir = Path(val_dir)

    assert dataset in ["cifar10", "cifar100", "stl10", "imagenet", "imagenet100", "custom"]

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            data_dir / train_dir,
            train=True,
            download=download,
            transform=T_train,
        )

        val_dataset = DatasetClass(
            data_dir / val_dir,
            train=True if validation else False,
            download=download,
            transform=T_val,
        )

    elif dataset == "stl10":
        train_dataset = STL10(
            data_dir / train_dir,
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = STL10(
            data_dir / val_dir,
            split="train" if validation else "test",
            download=download,
            transform=T_val,
        )

    elif dataset in ["imagenet", "imagenet100", "custom"]:
        train_dir = data_dir / train_dir
        val_dir = data_dir / val_dir

        train_dataset = ImageFolder(train_dir, T_train)
        val_dataset = ImageFolder(val_dir, T_val)

    return train_dataset, val_dataset


def prepare_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4, validation: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """
    if validation:
        N = len(train_dataset)
        idxs = list(range(N))
        N_train = int(N * 0.9)
        np.random.seed(1337)
        np.random.shuffle(idxs)
        train_idxs, valid_idxs = idxs[:N_train], idxs[N_train:]

        train_sampler = SubsetRandomSampler(train_idxs)
        valid_sampler = SubsetRandomSampler(valid_idxs)
        shuffle = None
    else:
        train_sampler = None
        valid_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=valid_sampler,
    )
    return train_loader, val_loader


def prepare_data(
    dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
    pretrain_augs: bool = False,
    validation: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        data_dir (Optional[Union[str, Path]], optional): path where to download/locate the dataset.
            Defaults to None.
        train_dir (Optional[Union[str, Path]], optional): subpath where the
            training data is located. Defaults to None.
        val_dir (Optional[Union[str, Path]], optional): subpath where the
            validation data is located. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader;.
    """

    T_train, T_val = prepare_transforms(dataset, pretrain_augs)
    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        download=download,
        validation=validation,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        validation=validation
    )
    return train_loader, val_loader
