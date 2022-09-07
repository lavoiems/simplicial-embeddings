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
from pprint import pprint

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import (WandbLogger, CSVLogger)
from orion.client import cli as orion_cli

from solo.args.setup import parse_args_pretrain
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer


try:
    from solo.methods.dali import PretrainABC
except ImportError as e:
    print(e)
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

import types

from solo.utils.checkpointer import Checkpointer
from solo.utils.classification_dataloader import prepare_data as prepare_data_classification
from solo.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
)


def main():
    args = parse_args_pretrain()
    seed_everything(args.seed)

    assert args.method in METHODS, f"Choose from {METHODS.keys()}"

    if args.num_large_crops != 2:
        assert args.method == "wmse"

    MethodClass = METHODS[args.method]
    if args.dali:
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with [dali]."
        MethodClass = types.new_class(f"Dali{MethodClass.__name__}", (PretrainABC, MethodClass))

    model = MethodClass(**args.__dict__)

    # pretrain dataloader
    if not args.dali:
        # asymmetric augmentations
        if args.unique_augs > 1:
            transform = [
                prepare_transform(args.dataset, **kwargs) for kwargs in args.transform_kwargs
            ]
        else:
            transform = [prepare_transform(args.dataset, **args.transform_kwargs)]

        transform = prepare_n_crop_transform(transform, num_crops_per_aug=args.num_crops_per_aug)
        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)

        train_dataset = prepare_datasets(
            args.dataset,
            transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            no_labels=args.no_labels,
        )
        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

    # normal dataloader for when it is available
    if args.dataset == "custom" and (args.no_labels or args.val_dir is None):
        val_loader = None
    elif args.dataset in ["imagenet100", "imagenet"] and args.val_dir is None:
        val_loader = None
    else:
        _, val_loader = prepare_data_classification(
            args.dataset,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path = None
    if args.auto_resume and args.resume_from_checkpoint is None:
        last_ckpt_dir = os.path.join(args.checkpoint_dir, 'last.ckpt')
        if os.path.exists(last_ckpt_dir):
            ckpt_path = last_ckpt_dir
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    callbacks = []

    # Logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.wandb:
        job_type = 'pretrain'
        logger = WandbLogger(
            name=args.name,
            save_dir=str(args.checkpoint_dir),
            offline=args.offline,
            resume="allow",
            id=args.name + '_' + job_type,
            job_type=job_type
        )
        logger.watch(model, log="gradients", log_freq=100)
        logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    else:
        logger = CSVLogger(save_dir=args.checkpoint_dir, name='pretrain')
        logger.log_hyperparams(args)

    if args.save_checkpoint:
        json_path = os.path.join(args.checkpoint_dir, "args.json")
        with open(json_path, 'w') as f:
            json.dump(vars(args), f, default=lambda o: "<not serializable>")

        ckpt = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='ep={epoch}',
            save_last=True, save_top_k=1,
            monitor=args.model_selection_score, mode='max',
            auto_insert_metric_name=False,
            save_weights_only=False,
            every_n_epochs=1, save_on_train_epoch_end=False
        )
        callbacks.append(ckpt)

    if args.auto_umap:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            args,
            logdir=os.path.join(args.auto_umap_dir, args.method),
            frequency=args.auto_umap_frequency,
        )
        callbacks.append(auto_umap)

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
    )

    try:
        if args.dali:
            trainer.fit(model, val_dataloaders=val_loader, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    except RuntimeError as e:
        orion_cli.report_bad_trial()
        print(e)
    else:
        # Orion minimize the following objective
        obj = 100 - float(trainer.callback_metrics[args.model_selection_score])
        orion_cli.report_objective(obj)


if __name__ == "__main__":
    main()
