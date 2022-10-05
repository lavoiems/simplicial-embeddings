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
import types

import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import (WandbLogger, CSVLogger)
from orion.client import cli as orion_cli

from solo.args.setup import parse_args_linear
from solo.methods import METHODS
from solo.utils.classification_dataloader import prepare_data

try:
    from solo.methods.dali import ClassificationABC, PretrainABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True


def main():
    args = parse_args_linear()
    seed_everything(args.seed)

    assert args.method in METHODS, f"Choose from {METHODS.keys()}"
    assert args.linear_method in METHODS, f"Choose from {METHODS.keys()}"

    MethodClass = METHODS[args.method]
    LinearModel = METHODS[args.linear_method]
    if args.dali:
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with [dali]."
        MethodClass = types.new_class(f"Dali{MethodClass.__name__}", (PretrainABC, MethodClass))
        LinearModel = types.new_class(f"Dali{LinearModel.__name__}", (ClassificationABC, LinearModel))

    args_path = str(args.pretrained_feature_extractor.parent / 'args.json')
    pretrain_args = json.load(open(args_path, 'r'))

    model = MethodClass(**pretrain_args)

    assert (
        args.pretrained_feature_extractor.match("*.ckpt")
        or args.pretrained_feature_extractor.match("*.pth")
        or args.pretrained_feature_extractor.match("*.pt")
    )
    ckpt_path = str(args.pretrained_feature_extractor)

    state = torch.load(ckpt_path)["state_dict"]
    for k in list(state.keys()):
        if "classifier" in k:
            del state[k]
    model.load_state_dict(state, strict=False)
    print(f"loaded {ckpt_path}")

    linear_model = LinearModel(model, **vars(args))

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
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_dir = str(args.checkpoint_dir)
    if args.wandb:
        job_type = 'train_linear'
        logger = WandbLogger(
            name=args.name,
            save_dir=checkpoint_dir,
            offline=args.offline,
            resume="allow",
            id=args.name + '_' + args.checkpoint_dir.name,
            job_type=job_type
        )
        logger.watch(linear_model, log="gradients", log_freq=100)
        logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    else:
        logger = CSVLogger(save_dir=checkpoint_dir, name='classifier')
        logger.log_hyperparams(args)

    if args.save_checkpoint:
        json_path = os.path.join(checkpoint_dir, "args.json")
        with open(json_path, 'w') as f:
            json.dump(vars(args), f, default=lambda o: "<not serializable>")

        ckpt = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='ep={epoch}',
            save_last=True, save_top_k=1,
            monitor=args.model_selection_score, mode='max',
            auto_insert_metric_name=False,
            save_weights_only=False,
            every_n_epochs=1, save_on_train_epoch_end=False
        )
        callbacks.append(ckpt)

    ckpt_path = None
    if args.auto_resume and args.resume_from_checkpoint is None:
        last_ckpt_dir = os.path.join(checkpoint_dir, 'last.ckpt')
        if os.path.exists(last_ckpt_dir):
            ckpt_path = last_ckpt_dir
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True) if args.accelerator == 'ddp' else None,
        enable_checkpointing=True,
    )

    try:
        if args.dali:
            trainer.fit(linear_model, val_dataloaders=val_loader, ckpt_path=ckpt_path)
        else:
            trainer.fit(linear_model, train_loader, val_loader, ckpt_path=ckpt_path)
    except RuntimeError as e:
        orion_cli.report_bad_trial()
        print(e)
    else:
        # Orion minimize the following objective
        obj = 100 - float(trainer.callback_metrics[args.model_selection_score])
        orion_cli.report_objective(obj)


if __name__ == "__main__":
    main()
