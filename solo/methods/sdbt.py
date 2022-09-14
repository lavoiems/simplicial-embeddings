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

import argparse
from typing import Any, List, Sequence, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseMethod
from solo.utils.metrics import accuracy_at_k, weighted_mean


class SoftmaxBridge(nn.Module):
    def __init__(self, message_size, voc_size, tau, **kwargs):
        super().__init__()
        self.message_size = message_size
        self.voc_size = voc_size
        self.tau = tau

    def forward(self, x, tau=None):
        logits = x.view(-1, self.message_size, self.voc_size)
        taus = tau or self.tau
        return F.softmax(logits / taus, -1).view(x.shape[0], -1)


class SDBarlowTwins(BaseMethod):
    def __init__(
        self, proj_hidden_dim: int, proj_output_dim: int, lamb: float, scale_loss: float,
        message_size: int,
        voc_size: int,
        taus: Sequence[float],
        tau_online: float,
        tau_target: float,
        **kwargs
    ):
        """Implements Barlow Twins (https://arxiv.org/abs/2103.03230)

        Args:
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            proj_output_dim (int): number of dimensions of projected features.
            lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
            scale_loss (float): scaling factor of the loss.
        """

        super().__init__(**kwargs)
        self.message_size = message_size
        self.voc_size = voc_size
        self.tau_online = tau_online
        self.tau_target = tau_target

        # Online
        self.embedder = nn.Sequential(
                             nn.Linear(self.features_dim, message_size*voc_size, bias=False),
                             nn.BatchNorm1d(message_size*voc_size)
                        )
        self.softmax = SoftmaxBridge(message_size, voc_size, tau_online, **kwargs)

        self.lamb = lamb
        self.scale_loss = scale_loss

        # projector
        self.projector = nn.Sequential(
            nn.Linear(message_size*voc_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # Y classifiers
        self.taus = taus
        self.linears_y = [torch.nn.Linear(message_size*voc_size, self.num_classes) for _ in range(len(self.taus))]
        self.linears_y = nn.ModuleList(self.linears_y)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SDBarlowTwins, SDBarlowTwins).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("barlow_twins")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--lamb", type=float, default=0.0051)
        parser.add_argument("--scale_loss", type=float, default=0.024)

        # Softmax bottleneck
        parser.add_argument("--taus", type=float, nargs='+', default=[1, 2])
        parser.add_argument("--voc_size", type=int, default=10)
        parser.add_argument("--message_size", type=int, default=100)
        parser.add_argument("--tau_online", type=float, default=0.3)
        parser.add_argument("--tau_target", type=float, default=5.0)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        extra_learnable_params = [{"name": f'classifier_y_{tau}',
              "params": linear_y.parameters(),
              "lr": self.classifier_lr,
              "weight_decay": 0
              }
              for tau, linear_y in zip(self.taus, self.linears_y)
        ]

        extra_learnable_params += [
            {"params": self.projector.parameters()},
            {"params": self.embedder.parameters()}
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X, *args, **kwargs):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        emb = self.embedder(out['feats'])
        y = self.softmax(emb)
        z = self.projector(y)
        return {**out, "z": z, "y": y, "emb": emb}

    def _class_step(self, X, targets, classifier):
        logits = classifier(X.detach())

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
        return {"loss": loss, "acc1": acc1, "acc5": acc5}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        *_, targets = batch
        class_loss = out["loss"]

        feats1, feats2 = out["feats"]

        emb1 = self.embedder(feats1)
        emb2 = self.embedder(feats2)
        y1 = self.softmax(emb1, tau=self.tau_online)
        y2 = self.softmax(emb2, tau=self.tau_target)
        z1 = self.projector(y1)
        z2 = self.projector(y2)

        # ----- y class loss
        emb = torch.cat([emb1, emb2]).detach().view(-1, self.message_size, self.voc_size)
        targets = torch.cat([targets, targets])
        outs_y = {tau: F.softmax(emb/tau, -1).view(emb.shape[0], -1) for tau in self.taus}
        online_class = [self._class_step(outs_y[tau], targets, linear_y) for tau, linear_y in zip(self.taus, self.linears_y)]
        online_class = {
            f"online_y_{tau}_" + k: v for tau, oc in zip(self.taus, online_class) for k, v in oc.items()
        }

        # ------- barlow twins loss -------
        barlow_loss = barlow_loss_func(z1, z2, lamb=self.lamb, scale_loss=self.scale_loss)

        online_class_loss = sum([online_class[f'online_y_{tau}_loss'] for tau in self.taus if not math.isnan(online_class[f'online_y_{tau}_loss'])])
        online_class = {
            "train_" + k: v for k, v in online_class.items()
        }

        metrics = {
            "train_barlow_loss": barlow_loss,
        }
        metrics.update(online_class)
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return barlow_loss + class_loss + online_class_loss

    @torch.no_grad()
    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validation step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding a batch of images in the momentum
        backbone and classifier and computing statistics.
        Args:
            batch (List[torch.Tensor]): a batch of data in the format of [X, Y].
            batch_idx (int): index of the batch.
        Returns:
            Tuple(Dict[str, Any], Dict[str, Any]): tuple of dicts containing the batch_size (used
                for averaging), the classification loss and accuracies for both the online and the
                momentum classifiers.
        """

        pm0 = super().validation_step(batch, batch_idx)

        x, targets = batch
        X = self.backbone(x)
        batch_size = targets.size(0)

        emb = self.embedder(X)

        emb = emb.view(-1, self.message_size, self.voc_size)

        outs_y = {tau: F.softmax(emb/tau, -1).view(emb.shape[0], -1) for tau in self.taus}
        online_class = [self._class_step(outs_y[tau], targets, linear_y) for tau, linear_y in zip(self.taus, self.linears_y)]
        online_class = {
            f"val_online_y_{tau}_" + k: v for tau, oc in zip(self.taus, online_class) for k, v in oc.items()
        }

        metrics = {
            "batch_size": batch_size,
        }

        metrics.update(online_class)

        return pm0, metrics

    def validation_epoch_end(self, outs: Tuple[List[Dict[str, Any]]]):
        """Averages the losses and accuracies of the momentum backbone / classifier for all the
        validation batches. This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        Args:
            outs (Tuple[List[Dict[str, Any]]]):): list of outputs of the validation step for self
                and the parent.
        """

        parent_outs = [out[0] for out in outs]
        super().validation_epoch_end(parent_outs)

        outs = [out[1] for out in outs]

        log = {k: weighted_mean(outs, k, 'batch_size') for k in outs[0].keys() if k != 'batch_size'}

        #online_val_loss = weighted_mean(outs, "val_online_loss", "batch_size")
        #online_val_acc1 = weighted_mean(outs, "val_online_acc1", "batch_size")
        #online_val_acc5 = weighted_mean(outs, "val_online_acc5", "batch_size")

        #log = {
        #    "online_val_y_loss": online_val_loss,
        #    "online_val_y_acc1": online_val_acc1,
        #    "online_val_y_acc5": online_val_acc5
        #}
        self.log_dict(log, sync_dist=True)

