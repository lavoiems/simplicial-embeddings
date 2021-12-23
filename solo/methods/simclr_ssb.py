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
from typing import Any, Dict, List, Sequence

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from solo.losses.simclr import simclr_loss_func
from solo.methods.simclr import SimCLR
from solo.methods.ssb import (ErasureChannel, SoftmaxBridge)


class SimCLRSSB(SimCLR):
    def __init__(self, message_size: int, voc_size: int, emdim: int,
                 pred_hidden_dim: int,
                 tau: float = 1., hard: bool = False, gumbel: bool = False,
                 channel_erase: float = 0., **kwargs):
        """Implements SimCLR+SSB (https://arxiv.org/abs/2002.05709).

        Args:
            TODO
        """
        kwargs['proj_output_dim'] = message_size * voc_size
        super().__init__(**kwargs)
        self.tau = tau
        self.gumbel = gumbel
        self.hard = hard
        self.erase = ErasureChannel(message_size, voc_size, channel_erase)
        self.bridge = SoftmaxBridge(message_size, voc_size, emdim)

        # predictor (could this be a transformer?)
        total_emdim = message_size * emdim
        self.predictor = nn.Sequential(
            nn.Linear(total_emdim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, total_emdim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimCLR, SimCLR).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("ssb")

        parser.add_argument('--message_size', default=128, type=int, help='message size')
        parser.add_argument('--voc_size', default=256, type=int, help='voc size')
        parser.add_argument('--emdim', default=8, type=int, help='embedded voc size')
        parser.add_argument("--pred_hidden_dim", type=int, default=512)
        parser.add_argument('--gumbel', choices=[True, False], type=eval, default=False)
        parser.add_argument('--hard', choices=[True, False], type=eval, default=False)
        parser.add_argument('--channel_erase', default=0., type=float, help='masking prob / token')
        parser.add_argument('--tau', default=1., type=float, help='softmax tau')

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.bridge.parameters()},
            {"params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X, *args, **kwargs)

        logits, z = self.bridge(out["z"],
                                erase=self.erase, tau=self.tau,
                                gumbel=self.gumbel, hard=self.hard)
        out["z"] = z.flatten(1)
        msg = logits.argmax(-1)
        y = F.one_hot(msg, num_classes=self.voc_size).float().view(z.size(0), -1)
        return {**out, "z": z, "y": y}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        feats = out["feats"]

        Z = [self.projector(f) for f in feats]
        z = torch.cat(Z)
        logits, z = self.bridge(z,
                                erase=self.erase, tau=self.tau,
                                gumbel=self.gumbel, hard=self.hard)
        msg = logits.argmax(-1)
        py = (msg.view(-1, 1) == torch.arange(self.hparams.voc_size, device=msg.device)).float().mean(0)
        perplexity = torch.exp(torch.log(py + 1e-8).mul(py).sum().neg())
        logpyx = F.log_softmax(logits, -1)
        hyx = logpyx.mul(logpyx.exp()).sum(-1).mean().neg() / math.log(2)  # entropy in bits
        #  y = F.one_hot(msg, num_classes=self.voc_size).float().view(y.size(0), -1)

        keys = torch.tensor_split(z, len(Z))
        queries = [self.predictor(key) for key in keys]
        # TODO

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        nce_loss = simclr_loss_func(
            z,
            indexes=indexes,
            temperature=self.temperature,
        )

        logs = {
            'train_nce_loss': nce_loss,
            'perplexity': perplexity,
            'hyx': hyx
        }
        self.log_dict(logs, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss
