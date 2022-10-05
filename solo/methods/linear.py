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

import math
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.methods.base import BaseMethod
from solo.utils.lars import LARSWrapper
from solo.utils.metrics import accuracy_at_k, weighted_mean
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
)


class LinearModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        max_epochs: int,
        total_max_epochs: Optional[int],
        batch_size: int,
        optimizer: str,
        lars: bool,
        lr: float,
        weight_decay: float,
        l1: float,
        exclude_bias_n_norm: bool,
        extra_optimizer_args: dict,
        scheduler: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        lr_decay_steps: Optional[Sequence[int]] = None,
        finetune: bool=False,
        **kwargs,
    ):
        """Implements linear evaluation.

        Args:
            model (nn.Module): pretrained model with all its modules, but
                classifiers freshly initialized.
            finetune (bool): whether we allow for supervised finetuning of the
                pretrained model (default is False).
            num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.
            batch_size (int): batch size.
            optimizer (str): optimizer to use.
            lars (bool): whether to use lars or not.
            lr (float): learning rate.
            weight_decay (float): weight decay.
            exclude_bias_n_norm (bool): whether to exclude bias and batch norm from weight decay
                and lars adaptation.
            extra_optimizer_args (dict): extra optimizer arguments.
            scheduler (str): learning rate scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            lr_decay_steps (Optional[Sequence[int]], optional): list of epochs where the learning
                rate will be decreased. Defaults to None.
        """

        super().__init__()

        self.model = model
        self.classifiers = nn.ModuleDict(
            {name: module
             for name, module in self.model.named_children()
             if 'classifier' in name}
        )
        self.extractors = nn.ModuleDict(
            {name: module
             for name, module in self.model.named_children()
             if 'classifier' not in name}
        )

        # training related
        self.max_epochs = max_epochs
        self.total_max_epochs = total_max_epochs or max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lars = lars
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1 = l1
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.lr_decay_steps = lr_decay_steps

        # all the other parameters
        self.extra_args = kwargs

        self.finetune = finetune
        self.model.requires_grad_(True)
        if not finetune:
            self.extractors.requires_grad_(False)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic linear arguments.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("linear")

        parser.add_argument("--finetune", action="store_true")

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--l1", type=float, default=0.0)
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")
        parser.add_argument("--total_max_epochs", type=int, default=None)

        # optimizer
        SUPPORTED_OPTIMIZERS = ["sgd", "adam", "adamw"]

        parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True)
        parser.add_argument("--lars", action="store_true")
        parser.add_argument("--exclude_bias_n_norm", action="store_true")

        # scheduler
        SUPPORTED_SCHEDULERS = [
            "reduce",
            "cosine",
            "warmup_cosine",
            "step",
            "exponential",
            "none",
        ]

        parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce")
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)

        return parent_parser

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        outs = self.model(X)
        logits = dict()
        l1_reg = 0.0
        for name, (feature_name, classifier_name) in self.model.named_classifiers.items():
            classifier = self.classifiers[classifier_name]
            logits[name] = classifier(outs[feature_name])
            if self.l1 > 0:
                l1_reg_ = sum(map(lambda x: x.absolute().sum(),
                                  classifier.parameters()), 0.)
                l1_reg = l1_reg + l1_reg_

        outs.update(logits=logits, l1_reg=l1_reg)
        return outs

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configures the optimizer for the linear layer.

        Raises:
            ValueError: if the optimizer is not in (sgd, adam).
            ValueError: if the scheduler is not in not in (warmup_cosine, cosine, reduce, step,
                exponential).

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")

        params = list(filter(lambda p: p.requires_grad, self.parameters()))

        optimizer = optimizer(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        if self.lars:
            optimizer = LARSWrapper(optimizer, exclude_bias_n_norm=self.exclude_bias_n_norm)

        # select scheduler
        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.total_max_epochs,
                warmup_start_lr=self.warmup_start_lr,
                eta_min=self.min_lr,
            )
        elif self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.total_max_epochs)
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

    def shared_step(
        self, X, target, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """
        output = self(X)
        logits = output['logits']

        outs = dict()
        for name, logits_ in logits.items():
            loss = F.cross_entropy(logits_, target, ignore_index=-1)
            acc1, acc5 = accuracy_at_k(logits_, target, top_k=(1, 5))
            outs_ = dict(loss=loss, acc1=acc1, acc5=acc5)
            outs.update(**{name + '_' + k: v for k, v in outs_.items()})

        outs['l1_reg'] = output['l1_reg']

        return outs

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set backbone to eval mode
        if not self.finetune:
            self.extractors.eval()

        X, target = batch
        outs = self.shared_step(X, target, batch_idx)
        loss = sum(map(lambda kv: kv[1],
                       filter(lambda kv: 'loss' in kv[0],
                              outs.items())), 0.0)
        l1_reg = outs['l1_reg']
        outs = {'train/' + k: v for k, v in outs.items()}
        self.log_dict(outs, on_epoch=True, sync_dist=True)

        return loss + self.l1 * l1_reg

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        X, target = batch
        batch_size = X.size(0)
        outs = self.shared_step(X, target, batch_idx)
        outs = {'val/' + k: v for k, v in outs.items()}
        outs['batch_size'] = batch_size
        return outs

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        metrics = {k: weighted_mean(outs, k, 'batch_size')
                   for k in outs[0].keys() if k != 'batch_size'}
        self.log_dict(metrics, sync_dist=True)
