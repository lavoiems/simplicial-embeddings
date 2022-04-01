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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from solo.losses.moco import moco_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.misc import gather
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


class SDMoCoV2Plus(BaseMomentumMethod):
    queue: torch.Tensor

    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        queue_size: int,
        message_size: int,
        voc_size: int,
        tau_online: float,
        tau_target: float,
        **kwargs
    ):
        """Implements MoCo V2+ (https://arxiv.org/abs/2011.10566).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            queue_size (int): number of samples to keep in the queue.
        """

        super().__init__(**kwargs)
        self.message_size = message_size
        self.voc_size = voc_size
        self.tau_online = tau_online
        self.tau_target = tau_target
        self.temperature = temperature
        self.queue_size = queue_size

        # projector
        self.embedder = nn.Sequential(
                            nn.Linear(self.features_dim, message_size*voc_size, bias=False),
                            nn.BatchNorm1d(message_size*voc_size))
        self.softmax = SoftmaxBridge(message_size, voc_size, tau_online, **kwargs)
        self.projector = nn.Sequential(
            nn.Linear(message_size*voc_size, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_embedder = nn.Sequential(
                            nn.Linear(self.features_dim, message_size*voc_size, bias=False),
                            nn.BatchNorm1d(message_size*voc_size))
        self.momentum_softmax = SoftmaxBridge(message_size, voc_size, tau_target, **kwargs)
        self.momentum_projector = nn.Sequential(
            nn.Linear(message_size*voc_size, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # create the queue
        self.register_buffer("queue", torch.randn(2, proj_output_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Y classifiers
        self.taus = [5e-2, 1e-1, 0.5, 1, 2]
        self.linears_y = [torch.nn.Linear(message_size*voc_size, self.num_classes) for _ in range(len(self.taus))]
        self.linears_y = nn.ModuleList(self.linears_y)
        #self.momentum_linear_y = torch.nn.Linear(message_size*voc_size, self.num_classes)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SDMoCoV2Plus, SDMoCoV2Plus).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("mocov2plus")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        # queue settings
        parser.add_argument("--queue_size", default=65536, type=int)

        # Softmax bottleneck
        parser.add_argument("--voc_size", type=int, default=10)
        parser.add_argument("--message_size", type=int, default=100)
        parser.add_argument("--tau_online", type=float, default=0.3)
        parser.add_argument("--tau_target", type=float, default=5.0)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters together with parent's learnable parameters.

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

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector), (self.embedder, self.momentum_embedder)]
        return super().momentum_pairs + extra_momentum_pairs

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner.

        Args:
            keys (torch.Tensor): output features of the momentum backbone.
        """

        batch_size = keys.shape[1]
        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(0, 2, 1)
        self.queue[:, :, ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr  # type: ignore

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the online backbone and projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        emb = self.embedder(out['feats'])
        y = self.softmax(emb)
        z = self.projector(y)
        z = F.normalize(z, dim=-1)
        return {**out, "z": z, "y": y, "emb": emb}

    def _class_step(self, X, targets, classifier):
        logits = classifier(X)

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
        return {"loss": loss, "acc1": acc1, "acc5": acc5}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for MoCo reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the
                format of [img_indexes, [X], Y], where [X] is a list of size self.num_large_crops
                containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MOCO loss and classification loss.

        """

        out = super().training_step(batch, batch_idx)
        *_, targets = batch
        class_z_loss = out["loss"]
        feats1, feats2 = out["feats"]
        momentum_feats1, momentum_feats2 = out["momentum_feats"]

        embs1 = self.embedder(feats1)
        Y1 = self.softmax(embs1)
        q1 = self.projector(Y1)
        embs2 = self.embedder(feats2)
        Y2 = self.softmax(embs2)
        q2 = self.projector(Y2)
        q1 = F.normalize(q1, dim=-1)
        q2 = F.normalize(q2, dim=-1)

        with torch.no_grad():
            k1 = self.momentum_softmax(self.momentum_embedder(momentum_feats1))
            k1 = self.momentum_projector(k1)
            k2 = self.momentum_softmax(self.momentum_embedder(momentum_feats2))
            k2 = self.momentum_projector(k2)
            k1 = F.normalize(k1, dim=-1)
            k2 = F.normalize(k2, dim=-1)

        # ------- contrastive loss -------
        # symmetric
        queue = self.queue.clone().detach()
        nce_loss = (
            moco_loss_func(q1, k2, queue[1], self.temperature)
            + moco_loss_func(q2, k1, queue[0], self.temperature)
        ) / 2

        # ------- update queue -------
        keys = torch.stack((gather(k1), gather(k2)))
        self._dequeue_and_enqueue(keys)

        with torch.no_grad():
            y = Y1.view(-1, self.message_size, self.voc_size)
            entropy = (y*torch.log(y)).neg().sum(-1)
            entropy = {
              'mu': entropy.mean(),
              'std': entropy.std()
            }

        emb = embs1.view(-1, self.message_size, self.voc_size)
        outs_y = {tau: F.softmax(emb/tau, -1).view(emb.shape[0], -1) for tau in self.taus}
        online_class = [self._class_step(outs_y[tau], targets, linear_y) for tau, linear_y in zip(self.taus, self.linears_y)]
        online_class = {
            f"online_y_{tau}_" + k: v for tau, oc in zip(self.taus, online_class) for k, v in oc.items()
        }

        online_class_loss = sum([online_class[f'online_y_{tau}_loss'] for tau in self.taus if not math.isnan(online_class[f'online_y_{tau}_loss'])])
        online_class = {
            "train_" + k: v for k, v in online_class.items()
        }

        metrics = {
            "train_nce_loss": nce_loss,
            "train_H_mu_nce": entropy['mu'],
            'train_H_std_nce': entropy['std'],
        }
        metrics.update(online_class)
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return nce_loss + class_z_loss + online_class_loss

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

        outs_y = {tau: F.softmax(emb/tau, -1).view(emb.shape[0], -1) for tau in self.taus}
        online_class = [self._class_step(outs_y[tau], targets, linear_y) for tau, linear_y in zip(self.taus, self.linears_y)]
        online_class = {
            f"val_online_y_{tau}_" + k: v for tau, oc in zip(self.taus, online_class) for k, v in oc.items()
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


        outs = [out[2] for out in outs]
        log = {k: weighted_mean(outs, k, 'batch_size') for k in outs[0].keys() if k != 'batch_size'}


        self.log_dict(log, sync_dist=True)
