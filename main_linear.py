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
import math

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torchvision.models import resnet18, resnet50, wide_resnet50_2
from solo.methods.linear_control import LinearModel

from solo.args.setup import parse_args_linear
from solo.methods.base import BaseMethod, wide_resnet50_4
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


class PartMM(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_parts: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PartMM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_parts = n_parts
        self.weight = nn.Parameter(torch.empty((n_parts, input_size, output_size), **factory_kwargs))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def extra_repr(self) -> str:
        return f'input_size={self.input_size}, output_size={self.output_size}, n_parts={self.n_parts}'

    def forward(self, x): # x: B x F
        # out = x.view(x.shape[0], -1, self.n_parts) # out: B x N x F/N | W: N x o/N x F/N
        # out = out.transpose(0, 2).contiguous() # out: N x F/N x B | W: N x o/N x F/N
        # out = torch.matmul(self.weight, out) # out: N x o/N x B
        # out = out.transpose(0, 2).contiguous().view(-1, self.output_size*self.n_parts) # out: B x o

        out = x.view(x.shape[0], self.n_parts, -1) # out: b x n x m/n
        out = out.transpose(0, 1).contiguous() # out: n x b x m/n || W: n x m/n x o/n
        out = torch.matmul(out, self.weight) # out: n x b x o/n
        out = out.transpose(0, 1).contiguous() # out: b x n x o/n
        out = out.view(x.shape[0], -1)

        return out


class Embedder(nn.Module):
    def __init__(self, features_dim, message_size, voc_size, n_parts=None, **kwargs):
        super().__init__()
        if not n_parts or n_parts == 1:
            self.embedder = nn.Sequential(
                                nn.Linear(features_dim, message_size*voc_size, bias=False),
                                nn.BatchNorm1d(message_size*voc_size))
        else:
            self.embedder = nn.Sequential(
                                PartMM(features_dim//n_parts, message_size*voc_size//n_parts, n_parts),
                                nn.BatchNorm1d(message_size*voc_size))

        self.message_size = message_size
        self.voc_size = voc_size

    def forward(self, x):
        o = self.embedder(x)
        return o.view(x.shape[0], self.message_size, self.voc_size)


class ICA(nn.Module):
    def __init__(self, feats_size: int, idx: int):
        super().__init__()
        self.feats_ica = nn.ModuleList([nn.Linear(feats_size, feats_size, bias=False) for _ in range(5)])
        self.idx = idx

    def forward(self, X: torch.Tensor):
        return self.feats_ica[self.idx](X)

def main():
    args = parse_args_linear()
    if args.linear_base:
        from solo.methods.linear import LinearModel
    elif args.mask:
        from solo.methods.linear_masked import LinearModel
    elif args.method == 'linear_fine':
        from solo.methods.linear_fine import LinearModelFine as LinearModel
    else:
        from solo.methods.linear_control import LinearModel

    assert args.backbone in BaseMethod._SUPPORTED_BACKBONES
    backbone_model = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "wide_resnet50_2": wide_resnet50_2,
        "wide_resnet50_4": wide_resnet50_4,
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

    args_path = os.path.join('/'.join(args.pretrained_feature_extractor.split('/')[:-1]), 'args.json')
    pretrain_args = json.load(open(args_path, 'r'))

    backbone = backbone_model(**kwargs)
    in_features = backbone.fc.in_features
    if "resnet" in args.backbone:
        # remove fc layer
        backbone.fc = nn.Identity()
        if cifar:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()

    assert (
        args.pretrained_feature_extractor.endswith(".ckpt")
        or args.pretrained_feature_extractor.endswith(".pth")
        or args.pretrained_feature_extractor.endswith(".pt")
    )
    ckpt_path = args.pretrained_feature_extractor

    state = torch.load(ckpt_path)["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            raise Exception(
                "You are using an older checkpoint."
                "Either use a new one, or convert it by replacing"
                "all 'encoder' occurances in state_dict with 'backbone'"
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        if 'embedder' not in k and 'feats_ica' not in k:
            del state[k]
    backbone.load_state_dict(state, strict=False)
    backbone.cuda()

    if not args.linear_base:
        if pretrain_args['method'] == 'byol':
            if args.ica:
                embedder = ICA(in_features, args.ica_idx)
                embedder.load_state_dict(state, strict=False)
            else:
                embedder = nn.Identity()
            pretrain_args['voc_size'] = 1
            pretrain_args['message_size'] = in_features
        else:
            embedder = Embedder(in_features, **pretrain_args)
            embedder.load_state_dict(state, strict=False)
        embedder.cuda()

    print(f"loaded {ckpt_path}")

    if args.dali:
        assert _dali_avaliable, "Dali is not currently avaiable, please install it first."
        Class = types.new_class(f"Dali{LinearModel.__name__}", (ClassificationABC, LinearModel))
    else:
        Class = LinearModel

    del args.backbone

    if not args.linear_base:
        model = Class(backbone, embedder, voc_size=pretrain_args['voc_size'], message_size=pretrain_args['message_size'], **args.__dict__)
    else:
        model = Class(backbone, **args.__dict__)

    train_loader, val_loader = prepare_data(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrain_augs=args.pretrain_augs,
        validation=args.validation,
    )

    callbacks = []

    # wandb logging
    #os.makedirs(args.checkpoint_dir, exist_ok=True)
    #os.makedirs(os.path.join(args.checkpoint_dir, 'wandb'), exist_ok=True)
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            #save_dir=args.checkpoint_dir if args.save_checkpoint else None,
            #group=args.group,
            entity=args.entity,
            offline=args.offline
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, "linear"),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    run_dir = os.path.dirname(ckpt_path)
    csv_logger = CSVLogger(
        save_dir=run_dir, name=f'classifier'
    )
    csv_logger.log_hyperparams(args)

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    if args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint
    else:
        ckpt_path = None

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else csv_logger,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True) if args.accelerator == 'ddp' else None,
        enable_checkpointing=False,
    )
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
