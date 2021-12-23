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
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.metrics import accuracy_at_k, weighted_mean


class SoftmaxBridge(nn.Module):
    def __init__(self, message_size, voc_size, tau, tau_noise=0, **kwargs):
        super().__init__()
        self.message_size = message_size
        self.voc_size = voc_size
        self.tau = tau
        self.tau_noise = tau_noise

    def forward(self, x):
        logits = x.view(-1, self.message_size, self.voc_size)

        if self.tau_noise > 0:
            tau_noise = torch.rand([logits.shape[0], logits.shape[1], 1],
                                    dtype=logits.dtype,
                                    device=logits.device)
            # Recenter, scale
            tau_noise = (tau_noise-0.5)*2*self.tau_noise
            taus = tau_noise.exp()*self.tau
        else:
            taus = self.tau

        return F.softmax(logits / taus, -1).view(x.shape[0], -1)


class SDBYOL(BaseMomentumMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        message_size: int,
        voc_size: int,
        tau_online: float,
        tau_target: float,
        **kwargs,
    ):
        """Implements BYOL (https://arxiv.org/abs/2006.07733).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(**kwargs)
        self.message_size = message_size
        self.voc_size = voc_size
        self.tau_online = tau_online
        self.tau_target = tau_target

        # Online
        self.embedder = nn.Sequential(
                            nn.Linear(self.features_dim, message_size*voc_size, bias=False),
                            nn.BatchNorm1d(message_size*voc_size))
        self.softmax = SoftmaxBridge(message_size, voc_size, tau_online, **kwargs)

        self.projector = nn.Sequential(
            nn.Linear(message_size*voc_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # Momentum
        self.momentum_embedder = nn.Sequential(
                            nn.Linear(self.features_dim, message_size*voc_size, bias=False),
                            nn.BatchNorm1d(message_size*voc_size))
        self.momentum_softmax = SoftmaxBridge(message_size, voc_size, tau_target, **kwargs)


        self.momentum_projector = nn.Sequential(
            nn.Linear(message_size*voc_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.embedder, self.momentum_embedder)
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        # Y classifiers
        self.linear_y = torch.nn.Linear(message_size*voc_size, self.num_classes)
        #self.momentum_linear_y = torch.nn.Linear(message_size*voc_size, self.num_classes)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SDBYOL, SDBYOL).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("byol")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)

        # Softmax bottleneck
        parser.add_argument("--voc_size", type=int, default=10)
        parser.add_argument("--message_size", type=int, default=100)
        parser.add_argument("--tau_online", type=float, default=0.3)
        parser.add_argument("--tau_target", type=float, default=5.0)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
            {"params": self.embedder.parameters()},
            {"name": 'classifier_y',
              "params": self.linear_y.parameters(),
              "lr": self.classifier_lr,
              "weight_decay": 0
              },
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector), (self.embedder, self.momentum_embedder)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        emb = self.embedder(out['feats'])
        y = self.softmax(emb)
        z = self.projector(y)
        p = self.predictor(z)
        return {**out, "z": z, "p": p, "logits": logits, "y": y}

    def _class_step(self, X, targets, classifier):
        logits = classifier(X)

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
        return {"loss": loss, "acc1": acc1, "acc5": acc5}

    def _shared_step(
        self, feats: List[torch.Tensor], momentum_feats: List[torch.Tensor], targets: List[torch.tensor]
    ) -> torch.Tensor:

        embs = [self.embedder(f) for f in feats]
        Y = [self.softmax(e) for e in embs]
        Z = [self.projector(y) for y in Y]
        P = [self.predictor(z) for z in Z]

        # forward momentum backbone
        with torch.no_grad():
            embs_momentum = [self.momentum_embedder(f) for f in momentum_feats]
            Y_momentum = [self.momentum_softmax(e) for e in embs_momentum]
            Z_momentum = [self.momentum_projector(y) for y in Y_momentum]

        # ------- negative consine similarity loss -------
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

        # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()

        emb = embs[0].view(-1, self.message_size, self.voc_size)
        y_hard = F.one_hot(emb.argmax(-1), num_classes=self.voc_size)
        y_hard = y_hard.view(y_hard.shape[0], -1)
        y_hard = y_hard.to(emb.dtype)
        online_class = self._class_step(y_hard, targets, self.linear_y)
        online_class = {
            "online_y_" + k: v for k, v in online_class.items()
        }

        return neg_cos_sim, z_std, online_class

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        *_, targets = batch

        neg_cos_sim, z_std, online_class = self._shared_step(out["feats"], out["momentum_feats"], targets)
        online_class_loss = online_class['online_y_loss']

        online_class = {
            "train_" + k: v for k, v in online_class.items()
        }

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }

        metrics.update(online_class)

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss + online_class_loss


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

        pm0, pm1 = super().validation_step(batch, batch_idx)

        x, targets = batch
        X = self.backbone(x)
        batch_size = targets.size(0)

        emb = self.embedder(X)

        emb = emb.view(-1, self.message_size, self.voc_size)
        y_hard = F.one_hot(emb.argmax(-1), num_classes=self.voc_size)
        y_hard = y_hard.view(y_hard.shape[0], -1)
        y_hard = y_hard.to(emb.dtype)
        online_class = self._class_step(y_hard, targets, self.linear_y)
        online_class = {
            "val_online_" + k: v for k, v in online_class.items()
        }


        metrics = {
            "batch_size": batch_size,
        }

        metrics.update(online_class)

        return pm0, pm1, metrics

    def validation_epoch_end(self, outs: Tuple[List[Dict[str, Any]]]):
        """Averages the losses and accuracies of the momentum backbone / classifier for all the
        validation batches. This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        Args:
            outs (Tuple[List[Dict[str, Any]]]):): list of outputs of the validation step for self
                and the parent.
        """

        parent_outs = [(out[0], out[1]) for out in outs]
        super().validation_epoch_end(parent_outs)

        if self.momentum_classifier is not None:
            outs = [out[2] for out in outs]

            online_val_loss = weighted_mean(outs, "val_online_loss", "batch_size")
            online_val_acc1 = weighted_mean(outs, "val_online_acc1", "batch_size")
            online_val_acc5 = weighted_mean(outs, "val_online_acc5", "batch_size")

            log = {
                "online_val_y_loss": online_val_loss,
                "online_val_y_acc1": online_val_acc1,
                "online_val_y_acc5": online_val_acc5
            }
            self.log_dict(log, sync_dist=True)

