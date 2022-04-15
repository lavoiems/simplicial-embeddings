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
        backbone: nn.Module,
        embedder: nn.Module,
        voc_size: int,
        message_size: int,
        num_classes: int,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lars: bool,
        lr: float,
        lrs: Sequence[float],
        wd1: Sequence[float],
        wd2: Sequence[float],
        taus: Sequence[float],
        eval_taus: Sequence[float],
        class_base: bool,
        weight_decay: float,
        exclude_bias_n_norm: bool,
        extra_optimizer_args: dict,
        scheduler: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        lr_decay_steps: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """Implements linear evaluation.

        Args:
            backbone (nn.Module): backbone architecture for feature extraction.
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

        self.backbone = backbone

        self.embedder = embedder
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        else:
            features_dim = self.backbone.num_features

        self.lrs = lrs
        self.wd1 = wd1
        self.wd2 = wd2
        self.taus = taus
        self.eval_taus = eval_taus
        self.class_base = class_base

        if self.class_base:
            self.classifiers = {}
            self.classifiers_base = {}
        self.classifiers_y = {}
        for lr in self.lrs:
            for wd1 in self.wd1:
                key = f'lr:{lr}_wd1:{wd1}_wd2:0'.replace('.', '')
                if self.class_base:
                    self.classifiers[key] = nn.Linear(features_dim, num_classes)
                    self.classifiers_base[key] = nn.Linear(voc_size*message_size, num_classes)

                classifiers_y = [nn.Linear(voc_size*message_size, num_classes) for _ in range(len(self.taus))]   # type: ignore
                classifiers_y = nn.ModuleList(classifiers_y)
                self.classifiers_y[key] = classifiers_y

            for wd2 in self.wd2:
                key = f'lr:{lr}_wd1:0_wd2:{wd2}'.replace('.', '')
                if self.class_base:
                    self.classifiers[key] = nn.Linear(features_dim, num_classes)
                    self.classifiers_base[key] = nn.Linear(voc_size*message_size, num_classes)

                classifiers_y = [nn.Linear(voc_size*message_size, num_classes) for _ in range(len(self.taus))]   # type: ignore
                classifiers_y = nn.ModuleList(classifiers_y)
                self.classifiers_y[key] = classifiers_y
        if self.class_base:
            self.classifiers = nn.ModuleDict(self.classifiers)
            self.classifiers_base = nn.ModuleDict(self.classifiers_base)
        self.classifiers_y = nn.ModuleDict(self.classifiers_y)

        # training related
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lars = lars
        self.lr = lr
        self.weight_decay = weight_decay
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.lr_decay_steps = lr_decay_steps

        # all the other parameters
        self.extra_args = kwargs

        # Freeze backbone components
        self.backbone.requires_grad_(False)
        self.embedder.requires_grad_(False)

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

        # backbone args
        parser.add_argument("--backbone", choices=BaseMethod._SUPPORTED_BACKBONES, type=str)
        # for ViT
        parser.add_argument("--patch_size", type=int, default=16)

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0)

        parser.add_argument("--lrs", type=float, nargs='+', default=[0.1, 0.05, 0.5])
        parser.add_argument("--wd1", type=float, nargs='+', default=[0, 1e-8, 1e-6, 1e-4])
        parser.add_argument("--wd2", type=float, nargs='+', default=[0, 1e-8, 1e-6])
        parser.add_argument("--class_base", type=eval, default=False)
        parser.add_argument("--taus", type=float, nargs='+', default=[0.5, 1, 2, 3, 5])
        parser.add_argument("--eval_taus", type=float, nargs='+', default=[0.5, 1, 2, 3, 5])
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--group", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        # optimizer
        SUPPORTED_OPTIMIZERS = ["sgd", "adam"]

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

        with torch.no_grad():
            feats = self.backbone(X)
            emb = self.embedder(feats)
        return {"feats": feats, "emb": emb}
        #logits = self.classifier(feats)
        #return {"logits": logits, "feats": feats}

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
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")

        params = []
        if self.class_base:
            for classifier in self.classifiers.values():
                params += [list(classifier.parameters())]
            for classifier_base in self.classifiers_base.values():
                params += [list(classifier_base.parameters())]

        for classifiers_y in self.classifiers_y.values():
            params += [list(classifier.parameters()) for classifier in classifiers_y]

        p = []
        for param in params:
            p += param

        optimizer = optimizer(
            p,
            lr=1,
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
                max_epochs=self.max_epochs,
                warmup_start_lr=self.warmup_start_lr,
                eta_min=self.min_lr,
            )
        elif self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.max_epochs)
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

    def get_metrics(self, feats, target, classifier, lr, wd1, wd2):
        out = classifier(feats)
        weight = classifier.weight
        loss = F.cross_entropy(out, target) + wd1*weight.absolute().sum() + wd2*weight.pow(2).sum()
        loss *= lr
        acc1 = accuracy_at_k(out, target, top_k=(1,))[0]
        return loss, acc1

    def shared_step(
        self, X, target, batch_idx: int, taus=None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """

        batch_size = X.size(0)

        outs = self(X)
        feats = outs['feats']
        emb = outs['emb']
        losses = {}
        accs1 = {}
        feats_tau = {}
        for lr in self.lrs:
            for wd1 in self.wd1:
                key = f'lr:{lr}_wd1:{wd1}_wd2:0'.replace('.', '')

                with torch.no_grad():
                    for tau1 in self.taus:
                        feats_tau[tau1] = {tau2: F.softmax(emb/tau2, -1).view(emb.shape[0], -1) for tau2 in (taus or [tau1,])}

                if self.class_base:
                    losses[f'loss_{key}_z'], accs1[f'acc1_{key}_z'] = self.get_metrics(feats, target, self.classifiers[key], lr, wd1, 0)
                    losses[f'loss_{key}_base'], accs1[f'acc1_{key}_base'] = self.get_metrics(emb.view(emb.shape[0], -1), target, self.classifiers_base[key], lr, wd1, 0)

                classifiers_y = self.classifiers_y[key]
                metrics_tau = {tau1: {tau2: self.get_metrics(v, target, classifier, lr, wd1, 0) for (tau2, v) in vs.items()} for (tau1, vs), classifier in zip(feats_tau.items(), classifiers_y)}
                for tau1, vs in metrics_tau.items():
                    for tau2, (loss, acc1) in vs.items():
                        losses[f'loss_{key}_tau:{tau1}_etau:{tau2}'] = loss
                        accs1[f'acc1_{key}_tau:{tau1}_etau:{tau2}'] = acc1

            for wd2 in self.wd2:
                key = f'lr:{lr}_wd1:0_wd2:{wd2}'.replace('.', '')

                with torch.no_grad():
                    for tau1 in self.taus:
                        feats_tau[tau1] = {tau2: F.softmax(emb/tau2, -1).view(emb.shape[0], -1) for tau2 in (taus or [tau1,])}

                if self.class_base:
                    losses[f'loss_{key}_z'], accs1[f'acc1_{key}_z'] = self.get_metrics(feats, target, self.classifiers[key], lr, 0, wd2)
                    losses[f'loss_{key}_base'], accs1[f'acc1_{key}_base'] = self.get_metrics(emb.view(emb.shape[0], -1), target, self.classifiers_base[key], lr, 0, wd2)

                classifiers_y = self.classifiers_y[key]
                metrics_tau = {tau1: {tau2: self.get_metrics(v, target, classifier, lr, 0, wd2) for (tau2, v) in vs.items()} for (tau1, vs), classifier in zip(feats_tau.items(), classifiers_y)}
                for tau1, vs in metrics_tau.items():
                    for tau2, (loss, acc1) in vs.items():
                        losses[f'loss_{key}_tau:{tau1}_etau:{tau2}'] = loss
                        accs1[f'acc1_{key}_tau:{tau1}_etau:{tau2}'] = acc1

        return batch_size, losses, accs1

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set backbone to eval mode
        self.backbone.eval()
        self.embedder.eval()  # There was a bug here.

        X, target = batch
        _, losses, accs1 = self.shared_step(X, target, batch_idx)
        losses = {f'train_{k}': v for k, v in losses.items()}
        accs1 = {f'train_{k}': v for k, v in accs1.items()}

        log = {}
        log.update(losses)
        log.update(accs1)
        self.log_dict(log, on_epoch=True, sync_dist=True)

        losses = losses.values()
        losses = [l for l in losses if not math.isnan(l)]
        loss = sum(losses)
        return loss

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
        batch_size, losses, accs1 = self.shared_step(X, target, batch_idx, taus=self.eval_taus)
        losses = {f'val_{k}': v for k, v in losses.items()}
        accs1 = {f'val_{k}': v for k, v in accs1.items()}

        results = {
            "batch_size": batch_size,
        }
        results.update(losses)
        results.update(accs1)
        return results

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        log = {k: weighted_mean(outs, k, 'batch_size') for k in outs[0].keys() if k != 'batch_size'}
        self.log_dict(log, sync_dist=True)
