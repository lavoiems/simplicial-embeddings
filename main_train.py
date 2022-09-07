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
import json

import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning import (Trainer, seed_everything)
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torchvision.models import resnet18, resnet50
from solo.methods.linear_control import LinearModel
from orion.client import cli as orion_cli

from solo.args.setup import parse_args_train
from solo.methods.base import BaseMethod
from solo.utils.backbones import (
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)

try:
    from solo.methods.dali import ClassificationABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True
import types

from solo.utils.checkpointer import Checkpointer
from solo.utils.classification_dataloader import prepare_data


class Embedder(nn.Module):
    def __init__(self, features_dim, message_size, voc_size, tau=1., **kwargs):
        super().__init__()
        self.embedder = nn.Sequential(
                            nn.Linear(features_dim, message_size*voc_size, bias=False),
                            nn.BatchNorm1d(message_size*voc_size))
        self.message_size = message_size
        self.voc_size = voc_size
        self.tau = tau

    def forward(self, x):
        o = self.embedder(x)
        o = o.view(-1, self.message_size, self.voc_size)
        return F.softmax(o / self.tau, dim =-1).flatten(1)


def main():
    args = parse_args_train()
    seed_everything(args.seed)
    from solo.methods.linear import LinearModel

    assert args.backbone in BaseMethod._SUPPORTED_BACKBONES
    backbone_model = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
    }[args.backbone]

    # initialize backbone
    kwargs = args.backbone_args
    cifar = kwargs.pop("cifar", False)
    # swin specific
    if "swin" in args.backbone and cifar:
        kwargs["window_size"] = 4

    backbone = backbone_model(**kwargs)
    features_dim = backbone.fc.in_features
    if "resnet" in args.backbone:
        # remove fc layer
        backbone.fc = nn.Identity()
        if cifar:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()

    if args.use_sem:
        embedder = Embedder(features_dim, **vars(args))
        features_dim = args.message_size * args.voc_size
    else:
        embedder = nn.Identity()
    backbone = nn.Sequential(backbone, embedder)

    if args.dali:
        assert _dali_avaliable, "Dali is not currently avaiable, please install it first."
        Class = types.new_class(f"Dali{LinearModel.__name__}", (ClassificationABC, LinearModel))
    else:
        Class = LinearModel

    args.finetune = True
    del args.backbone
    model = Class(backbone, features_dim, **vars(args))

    train_loader, val_loader = prepare_data(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrain_augs=args.pretrain_augs,
    )

    callbacks = []

    # wandb logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.wandb:
        os.makedirs(os.path.join(args.checkpoint_dir, 'wandb'), exist_ok=True)
        wandb_logger = WandbLogger(
            name=args.name,
            id=args.name,
            save_dir=args.checkpoint_dir,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            group=args.group,
            job_type='train'
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    else:
        csv_logger = CSVLogger(save_dir=args.checkpoint_dir, name='pretrain')
        csv_logger.log_hyperparams(args)

    if args.save_checkpoint:
        json_path = os.path.join(args.checkpoint_dir, "args.json")
        with open(json_path, 'w') as f:
            json.dump(vars(args), f, default=lambda o: "<not serializable>")

        ckpt = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='ep={epoch}',
            save_last=True, save_top_k=1,
            monitor='val_acc1', mode='max',
            auto_insert_metric_name=False,
            save_weights_only=False,
            every_n_epochs=args.checkpoint_frequency,
            save_on_train_epoch_end=False
        )
        callbacks.append(ckpt)

    ckpt_path = None
    if args.auto_resume and args.resume_from_checkpoint is None:
        last_ckpt_dir = os.path.join(args.checkpoint_dir, 'last.ckpt')
        if os.path.exists(last_ckpt_dir):
            ckpt_path = last_ckpt_dir
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else csv_logger,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True) if args.accelerator == 'ddp' else None,
        enable_checkpointing=True,
    )

    try:
        if args.dali:
            trainer.fit(model, val_dataloaders=val_loader, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    except RuntimeError:
        orion_cli.report_bad_trial()
        raise
    else:
        # Orion minimize the following objective
        obj = 100 - float(trainer.callback_metrics["val_acc1"])
        orion_cli.report_objective(obj)


if __name__ == "__main__":
    main()
