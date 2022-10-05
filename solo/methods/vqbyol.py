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
from typing import Any, Dict, Text, List, Sequence, Tuple
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.metrics import accuracy_at_k, weighted_mean


class VectorQuantizer(nn.Module):
    def __init__(self,
                 code_length: int, code_options: int,
                 shared_codebook: bool=True,
                 vq_hard: bool=True,
                 vq_ladder: bool=False,
                 vq_gumbel: bool=False,
                 vq_tau: float=1.0,
                 vq_dimz: float=8,
                 vq_beta: float=0.25, vq_eps: float = 1e-8,
                 vq_update_rule: Text = 'loss',
                 vq_spherical: bool=False,
                 vq_ema: float = 0.95):
        super().__init__()
        self.code_length = code_length
        self.code_options = code_options
        assert(not vq_hard or (vq_hard and not vq_ladder))
        assert(vq_tau >= 0.0)
        self.hard = vq_hard
        self.ladder = vq_ladder
        self.gumbel = vq_gumbel
        self.tau = vq_tau
        assert(vq_dimz is not None)
        self.dimz = vq_dimz
        self.beta = vq_beta
        self.eps = vq_eps
        self.spherical = vq_spherical

        self.update_rule = vq_update_rule
        self.ema = vq_ema
        self.register_buffer('has_init', torch.tensor(False, dtype=torch.bool))

        self.shared_codebook = shared_codebook
        if shared_codebook:
            self.codebook = nn.Parameter(torch.zeros(code_options, vq_dimz))
        else:
            self.codebook = nn.Parameter(torch.zeros(code_length, code_options, vq_dimz))

        if not bool(self.has_init):
            self.init(code_length)
            self.has_init.fill_(True)

    def init(self, code_dims):
        torch.nn.init.normal_(self.codebook,
                              mean=0.0, std=np.sqrt(2./(code_dims * self.dimz)))
        if self.spherical:
            with torch.no_grad():
                self.codebook.copy_(F.normalize(self.codebook, p=2, dim=-1))
        # torch.nn.init.uniform_(self.codebook, a=-1.0 / self.code_options, b=1.0 / self.code_options)

    def extra_repr(self):
        return f'L: {self.code_length}, V: {self.code_options}, dimz: {self.dimz}, update: {self.update_rule}'

    def latent(self, query):
        return query.view(-1, self.code_length, self.dimz)

    def score(self, latent):
        latent = latent[..., None, :]  # (B, code.length, code.options, dimz)
        if self.spherical:
            scores = F.cosine_similarity(latent, self.codebook, dim=-1) / self.tau
        else:
            scores = - 0.5 * latent.sub(self.codebook).pow(2).sum(-1) / self.tau
        return scores  # (B, code.length, code.options)

    @torch.no_grad()
    def ema_codebook_update(self, latent, y, ema=None):
        ema = ema or self.ema
        latent = latent[..., None, :]  # (B, ..., code.options, dimz)
        idx = y[..., None] == torch.arange(self.code_options, device=latent.device)  # (B, ..., code.options)
        if self.shared_codebook:
            batch_dims = tuple(range(idx.dim() - 1))
        else:
            batch_dims = 0
        new_codebook = latent.masked_fill(idx[..., None], 0.).sum(dim=batch_dims).div(
            idx.sum(dim=batch_dims).clamp(min=1.).unsqueeze_(-1))
        p_y = idx.sum(dim=batch_dims) / idx.sum()
        diff_codebook = new_codebook.sub(self.codebook)
        self.codebook.add_(diff_codebook.mul(1. - ema))
        if self.spherical:
            self.codebook.copy_(F.normalize(self.codebook, p=2, dim=-1))
        return diff_codebook.pow(2).sum(-1).max(), p_y

    def encode(self, latent):
        scores = self.score(latent)  # (B, ..., code.optionsn)
        if self.gumbel or self.ladder:
            return F.gumbel_softmax(scores, dim=-1)
        return F.softmax(scores, dim=-1)

    def decode(self, latent, y):
        if self.hard:
            code = y.argmax(-1)  # (B, ...)
        else:
            code = y
        if self.shared_codebook:
            if self.hard:
                values = self.codebook[code]  # (B, ..., dimz)
            else:
                values = torch.einsum('blv,vz->blz', code, self.codebook)
        else:
            if self.hard:
                values = self.codebook[torch.arange(self.code_length)[None, :], code]  # (B, ..., dimz)
            else:
                values = torch.einsum('blv,lvz->blz', code, self.codebook)
        if self.ladder:
            beta = self.beta**0.5
            values = (1 - beta) * values + beta * latent + torch.randn_like(values) * self.tau**0.5
        return values

    def forward(self, query):
        latent = self.latent(query)  # (B, ..., dimz)
        y = self.encode(latent)
        quantized = self.decode(latent, y)  # (B, ..., dimz)

        ema_update = self.update_rule == 'ema'
        if self.hard:
            embedding = (latent - latent.detach()) + (quantized.detach() if ema_update else quantized)
        else:
            embedding = quantized

        loss = 0.
        if self.training:
            if self.hard:
                loss = self.beta * quantized.detach().sub(latent).pow(2).sum(-1).mean()
                if self.update_rule == 'loss':
                    loss = loss + quantized.sub(latent.detach()).pow(2).sum(-1).mean()
                elif ema_update:
                    self.ema_codebook_update(latent, y)
            else:
                if self.spherical:
                    logNs = - F.cosine_similarity(quantized[:, :, None, :], self.codebook, dim=-1)
                else:
                    logNs = 0.5 * quantized[:, :, None, :].sub(self.codebook).pow(2).sum(-1)
                loss = (self.beta / self.tau) * torch.einsum('blv,blv->bl', y, logNs).mean()

        py = y.mean(dim=tuple(range(y.dim()-1)))
        Hy = - torch.sum(py * torch.log2(py + 1e-6))
        Hyx = - torch.mean(y.mul(torch.log2(y + 1e-6)).sum(-1))
        if self.hard:
            y = F.one_hot(y.argmax(-1), num_classes=self.code_options).to(dtype=query.dtype)
        return {'logits': latent, 'embedding': embedding.flatten(1), 'msg': y.argmax(-1),
                'representation': y.flatten(1),
                'metrics': {'Hy': Hy, 'Hyx': Hyx},
                'loss': loss}


class VQBYOL(BaseMomentumMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        message_size: int,
        voc_size: int,
        classifier_lasso: float,
        shared_codebook: bool=True,
        vq_hard: bool=True,
        vq_ladder: bool=False,
        vq_gumbel: bool=False,
        vq_spherical: bool=False,
        vq_tau: float=1.0,
        vq_dimz: float=8,
        vq_beta: float=0.25, vq_eps: float=1e-8,
        vq_update_rule: Text='loss',
        vq_ema: float=0.95,
        vq_code_wd: float=0.,
        vq_rel_loss: float=1.,
        **kwargs,
    ):
        """Implements BYOL (https://arxiv.org/abs/2006.07733).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """
        super().__init__(**kwargs)
        self.classifier_lasso = classifier_lasso

        self.message_size = message_size
        self.voc_size = voc_size
        self.vq_code_wd = vq_code_wd
        self.vq_rel_loss = vq_rel_loss

        # Online
        self.embedder = nn.Sequential(
             nn.Linear(self.features_dim, message_size*vq_dimz, bias=False),
             nn.BatchNorm1d(message_size*vq_dimz)
        )
        self.vq = VectorQuantizer(message_size, voc_size,
            shared_codebook=shared_codebook,
            vq_tau=vq_tau, vq_hard=vq_hard, vq_ladder=vq_ladder, vq_gumbel=vq_gumbel,
            vq_dimz=vq_dimz, vq_beta=vq_beta,
            vq_eps=vq_eps, vq_update_rule=vq_update_rule,
            vq_ema=vq_ema, vq_spherical=vq_spherical)

        self.projector = nn.Sequential(
            nn.Linear(message_size*vq_dimz, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # Momentum
        self.momentum_embedder = deepcopy(self.embedder)
        initialize_momentum_params(self.embedder, self.momentum_embedder)
        self.momentum_projector = deepcopy(self.projector)
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        # Y classifiers
        self.classifier_y = nn.Linear(message_size*voc_size, self.num_classes)
        self.classifier_vq = nn.Linear(message_size*vq_dimz, self.num_classes)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(VQBYOL, VQBYOL).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("byol")

        parser.add_argument("--classifier_lasso", type=float, default=0.)

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)

        # VQ bottleneck
        parser.add_argument("--voc_size", type=int, default=10)
        parser.add_argument("--message_size", type=int, default=100)
        parser.add_argument("--shared_codebook", type=eval,
            choices=[True, False], default=True)
        parser.add_argument("--vq_hard", type=eval,
            choices=[True, False], default=True)
        parser.add_argument("--vq_ladder", type=eval,
            choices=[True, False], default=False)
        parser.add_argument("--vq_gumbel", type=eval,
            choices=[True, False], default=False)
        parser.add_argument("--vq_spherical", type=eval,
            choices=[True, False], default=False)
        parser.add_argument("--vq_tau", type=float, default=1.0)
        parser.add_argument("--vq_dimz", type=int, default=8)
        parser.add_argument("--vq_beta", type=float, default=0.25)
        parser.add_argument("--vq_eps", type=float, default=1e-8)
        parser.add_argument("--vq_update_rule", type=str,
            choices=['loss', 'ema'], default='loss')
        parser.add_argument("--vq_ema", type=float, default=0.95)
        parser.add_argument("--vq_code_wd", type=float, default=0.)
        parser.add_argument("--vq_rel_loss", type=float, default=1.)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": f'classifier_y',
             "params": self.classifier_y.parameters(),
             "lr": self.classifier_lr,
             "weight_decay": 0.,
            },
            {"name": f'classifier_vq',
             "params": self.classifier_vq.parameters(),
             "lr": self.classifier_lr,
             "weight_decay": self.classifier_wd,
            },
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
            {"params": self.embedder.parameters()},
            {"params": self.vq.parameters(),
             "weight_decay": self.vq_code_wd,
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

    def forward_(self, feats):
        emb = self.embedder(feats)
        vq_outs = self.vq(emb)
        vq_emb = vq_outs['embedding']
        y = vq_outs['representation']
        z = self.projector(vq_emb)
        p = self.predictor(z)
        return {"z": z, "p": p, "y": y, "emb": vq_emb,
                'vq_loss': vq_outs['loss'], 'vq_metrics': vq_outs['metrics']}

    def momentum_forward_(self, feats):
        emb = self.momentum_embedder(feats)
        vq_outs = self.vq(emb)
        vq_emb = vq_outs['embedding'].detach()
        z = self.momentum_projector(vq_emb)
        return {"z": z}

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        feats = out['feats']
        out2 = self.forward_(feats)
        out.update(out2)
        return out

    @property
    def named_classifiers(self):
        classifiers = super().named_classifiers
        classifiers.update(**dict(y=('y', 'classifier_y'),
                                  vq=('emb', 'classifier_vq')))
        return classifiers

    def _class_step(self, X, targets, classifier):
        logits = classifier(X.detach())
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        l1_loss = sum(v.absolute().sum() for v in classifier.parameters())
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
        return {"loss": loss, "l1_loss": l1_loss, "acc1": acc1, "acc5": acc5}

    def _shared_step(
        self, feats: List[torch.Tensor], momentum_feats: List[torch.Tensor], targets: List[torch.tensor]
    ) -> torch.Tensor:

        online_outs = [self.forward_(f) for f in feats]
        online_outs = {k: [out[k] for out in online_outs] for k in online_outs[0].keys()}
        with torch.no_grad():
            mom_outs = [self.momentum_forward_(f) for f in momentum_feats]
            mom_outs = {k: [out[k] for out in mom_outs] for k in mom_outs[0].keys()}

        # ------- negative consine similarity loss -------
        P = online_outs['p']
        mom_Z = mom_outs['z']
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                neg_cos_sim += byol_loss_func(P[v2], mom_Z[v1])

        vq_loss = sum(online_outs['vq_loss']) / self.num_crops

        metrics = online_outs['vq_metrics'][0]
        metrics = {'vq_' + k: v for k, v in metrics.items()}

        # calculate std of features
        Z = online_outs['z']
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[:self.num_large_crops]), dim=-1).std(dim=1).mean()

        Y = online_outs['y'][:self.num_large_crops]
        y_class_outs = [self._class_step(y, targets, self.classifier_y) for y in Y]
        y_class_outs = {k: [out[k] for out in y_class_outs] for k in y_class_outs[0].keys()}
        y_class_outs = {'online_y_' + k: sum(y_class_outs[k]) / self.num_large_crops
                        for k in y_class_outs.keys()}

        emb = online_outs['emb'][:self.num_large_crops]
        vq_class_outs = [self._class_step(y, targets, self.classifier_vq) for y in emb]
        vq_class_outs = {k: [out[k] for out in vq_class_outs] for k in vq_class_outs[0].keys()}
        vq_class_outs = {'online_vq_' + k: sum(vq_class_outs[k]) / self.num_large_crops
                         for k in vq_class_outs.keys()}

        metrics.update(y_class_outs)
        metrics.update(vq_class_outs)

        online_class_loss = y_class_outs['online_y_loss'] + vq_class_outs['online_vq_loss']
        online_class_loss = online_class_loss + self.classifier_lasso * y_class_outs['online_y_l1_loss']

        return neg_cos_sim, vq_loss, online_class_loss, metrics

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

        neg_cos_sim, vq_loss, online_class_loss, metrics = self._shared_step(out["feats"], out["momentum_feats"], targets)

        metrics = {
            "train/" + k: v for k, v in metrics.items()
        }
        metrics["train/neg_cos_sim"] = neg_cos_sim
        metrics["train/vq_loss"] = vq_loss

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        total_loss = neg_cos_sim + self.vq_rel_loss * vq_loss + class_loss + online_class_loss
        if torch.isnan(total_loss):
            raise RuntimeError('nan value')
        return total_loss

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

        outs = self.forward_(X)
        y = outs['y']
        y_class_outs = self._class_step(y, targets, self.classifier_y)
        y_class_outs = {'val/online_y_' + k: v for k, v in y_class_outs.items()}

        emb = outs['emb']
        vq_class_outs = self._class_step(emb, targets, self.classifier_vq)
        vq_class_outs = {'val/online_vq_' + k: v for k, v in vq_class_outs.items()}


        metrics = {
            "batch_size": batch_size,
        }
        metrics.update(y_class_outs)
        metrics.update(vq_class_outs)

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

        vq_outs = [out[2] for out in outs]
        metrics = {k: weighted_mean(vq_outs, k, 'batch_size') for k in vq_outs[0].keys() if k != 'batch_size'}
        self.log_dict(metrics, sync_dist=True)

