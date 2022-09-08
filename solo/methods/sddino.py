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
import math
import distutils
from typing import Any, List, Sequence, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.dino import DINOLoss
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.misc import trunc_normal_
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


class DINOHead(nn.Module):
    mlp: Any
    last_layer: Any

    def __init__(
        self,
        in_dim: int,
        num_prototypes: int,
        use_bn: bool = True,
        norm_last_layer: bool = True,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        """DINO head that takes as input the features of the backbone, projects them in a lower
        dimensional space and multiplies with the prototypes.

        Args:
            in_dim (int): number of dimensions of the input (aka backbone features).
            num_prototypes (int): number of prototypes.
            use_bn (bool, optional): whether to use batch norm in projector. Defaults to True.
            norm_last_layer (bool, optional): whether to l2-norm the last layer. Defaults to True.
            num_layers (int, optional): number of layers in projector. Defaults to 3.
            hidden_dim (int, optional): number of dimension in hidden layers. Defaults to 2048.
            bottleneck_dim (int, optional): number of dimensions in bottleneck. Defaults to 256.
        """

        super().__init__()

        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: List[Any] = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, num_prototypes, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)  # type: ignore

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m: nn.Module):
        """Initializes weights with truncated normal and biases with zeros.

        Args:
            m (nn.Module): a layer of the DINO head.
        """

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the backbone, the projector and the last layer (prototypes).

        Args:
            x (torch.Tensor): a batch of features.

        Returns:
            torch.Tensor: a batch of logits.
        """

        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class SDDINO(BaseMomentumMethod):
    def __init__(
        self,
        proj_hidden_dim: int,
        proj_output_dim: int,
        num_prototypes: int,
        use_bn_in_head: bool,
        norm_last_layer: bool,
        clip_grad: float,
        freeze_last_layer: bool,
        student_temperature: float,
        teacher_temperature: float,
        warmup_teacher_temperature: float,
        warmup_teacher_temperature_epochs: int,
        message_size: int,
        voc_size: int,
        taus: Sequence[float],
        tau_online: float,
        tau_target: float,
        **kwargs,
    ):
        """Adds DINO head to the student and momentum DINO head to the teacher.

        Args:
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            proj_output_dim (int): number of output neurons in the projector.
            num_prototypes (int): number of prototypes.
            use_bn_in_head (bool): whether or not to use bn in the head.
            norm_last_layer (bool): whether or not to normalize the last layer (prototypes).
            clip_grad (float): threshold for gradient clipping.
            freeze_last_layer (bool): whether or not to freeze the last layer (prototypes).
            student_temperature (float): temperature for the student.
            teacher_temperature (float): temperature for the teacher.
            warmup_teacher_temperature (float): base temperature for the teacher.
            warmup_teacher_temperature_epochs (int): number of epochs of cosine annealing
                scheduling for teacher temperature.
        """

        super().__init__(**kwargs)

        self.clip_grad = clip_grad
        self.freeze_last_layer = freeze_last_layer

        self.embedder = nn.Sequential(
                             nn.Linear(self.features_dim, message_size*voc_size, bias=False),
                             nn.BatchNorm1d(message_size*voc_size)
                        )
        self.softmax = SoftmaxBridge(message_size, voc_size, tau_online, **kwargs)

        # dino head
        self.head = DINOHead(
            in_dim=message_size*voc_size,
            hidden_dim=proj_hidden_dim,
            use_bn=use_bn_in_head,
            bottleneck_dim=proj_output_dim,
            num_prototypes=num_prototypes,
            norm_last_layer=norm_last_layer,
        )

        self.momentum_embedder = nn.Sequential(
                             nn.Linear(self.features_dim, message_size*voc_size, bias=False),
                             nn.BatchNorm1d(message_size*voc_size)
                        )
        self.momentum_softmax = SoftmaxBridge(message_size, voc_size, tau_target, **kwargs)

        # instantiate and initialize momentum dino head
        self.momentum_head = DINOHead(
            in_dim=message_size*voc_size,
            hidden_dim=proj_hidden_dim,
            use_bn=use_bn_in_head,
            bottleneck_dim=proj_output_dim,
            num_prototypes=num_prototypes,
            norm_last_layer=norm_last_layer,
        )
        initialize_momentum_params(self.embedder, self.momentum_embedder)
        initialize_momentum_params(self.head, self.momentum_head)

        # dino loss
        self.dino_loss_func = DINOLoss(
            num_prototypes=num_prototypes,
            student_temp=student_temperature,
            warmup_teacher_temp=warmup_teacher_temperature,
            teacher_temp=teacher_temperature,
            warmup_teacher_temp_epochs=warmup_teacher_temperature_epochs,
            num_epochs=self.max_epochs,
        )

        self.message_size = message_size
        self.voc_size = voc_size
        self.taus = taus
        self.linears_y = [torch.nn.Linear(message_size*voc_size, self.num_classes) for _ in range(len(self.taus))]
        self.linears_y = nn.ModuleList(self.linears_y)
        self.linears_y = nn.ModuleList(self.linears_y)
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SDDINO, SDDINO).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("dino")

        # optimization settings
        parser.add_argument("--clip_grad", type=float, default=0)
        parser.add_argument("--freeze_last_layer", type=int, default=1)

        # dino head
        parser.add_argument("--proj_output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--num_prototypes", type=int, default=4096)
        parser.add_argument("--norm_last_layer", type=distutils.util.strtobool, default=True)
        parser.add_argument("--use_bn_in_head", type=distutils.util.strtobool, default=False)

        # temperature settings
        parser.add_argument("--student_temperature", type=float, default=0.1)
        parser.add_argument("--teacher_temperature", default=0.07, type=float)
        parser.add_argument("--warmup_teacher_temperature", default=0.04, type=float)
        parser.add_argument("--warmup_teacher_temperature_epochs", default=50, type=int)
        parser.add_argument("--voc_size", default=13, type=int)
        parser.add_argument("--message_size", default=5000, type=int)
        parser.add_argument("--tau_online", default=1, type=float)
        parser.add_argument("--tau_target", default=1, type=float)
        parser.add_argument("--taus", type=float, nargs='+', default=[1, 2])

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds DINO head parameters to the parent's learnable parameters.

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
        extra_learnable_params += [{"params": self.head.parameters()}]
        extra_learnable_params += [{"params": self.embedder.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (head, momentum_head) to the parent's momentum pairs.

        Returns:
            List[dict]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.head, self.momentum_head), (self.embedder, self.momentum_embedder)]
        return super().momentum_pairs + extra_momentum_pairs

    def dino_clip_gradients(self, clip: float):
        """Clips gradients after backward pass.

        Args:
            clip (float): threshold for gradient clipping.
        """

        for p in self.backbone.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)

    def on_train_epoch_start(self):
        """Updates the current epoch in DINO's loss object."""
        self.dino_loss_func.epoch = self.current_epoch

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the student (backbone and head).

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the logits of the head.
        """

        out = super().forward(X, *args, **kwargs)
        emb = self.embedder(out['feats'])
        y = self.softmax(emb)
        z = self.head(y)
        return {**out, "z": z, "y": y, "emb": emb}


    def _class_step(self, X, targets, classifier):
        logits = classifier(X.detach())

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
        return {"loss": loss, "acc1": acc1, "acc5": acc5}


    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for DINO reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where [X]
                is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of DINO loss and classification loss.
        """

        *_, targets = batch
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]
        momentum_feats1, momentum_feats2 = out["momentum_feats"]

        # forward online backbone
        emb1 = self.embedder(feats1)
        emb2 = self.embedder(feats2)
        y1 = self.softmax(emb1)
        y2 = self.softmax(emb2)
        p1 = self.head(y1)
        p2 = self.head(y2)
        p = torch.cat((p1, p2))

        # forward momentum backbone
        emb1_momentum = self.momentum_embedder(momentum_feats1)
        emb2_momentum = self.momentum_embedder(momentum_feats2)
        y1_momentum = self.momentum_softmax(emb1_momentum)
        y2_momentum = self.momentum_softmax(emb2_momentum)
        p1_momentum = self.momentum_head(y1_momentum)
        p2_momentum = self.momentum_head(y2_momentum)
        p_momentum = torch.cat((p1_momentum, p2_momentum))

        # ------- contrastive loss -------
        dino_loss = self.dino_loss_func(p, p_momentum)
        metrics = {"dino_loss": dino_loss}


        with torch.no_grad():
            emb = emb1.view(-1, self.message_size, self.voc_size)
            outs_y = {tau: F.softmax(emb/tau, -1).view(emb.shape[0], -1) for tau in self.taus}
        online_class = [self._class_step(outs_y[tau], targets, linear_y) for tau, linear_y in zip(self.taus, self.linears_y)]
        online_class = {
            f"online_y_{tau}_" + k: v for tau, oc in zip(self.taus, online_class) for k, v in oc.items()
        }
        online_class_loss = sum([online_class[f'online_y_{tau}_loss'] for tau in self.taus if not math.isnan(online_class[f'online_y_{tau}_loss'])])

        metrics.update(online_class)

        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return dino_loss + class_loss + online_class_loss

    def on_after_backward(self):
        """Performs gradient clipping and zeros the gradients on the last layer (prototypes)."""

        # clip gradients
        if self.clip_grad:
            self.dino_clip_gradients(self.clip_grad)
        # zero gradients on last layer
        if self.current_epoch < self.freeze_last_layer:
            for p in self.head.last_layer.parameters():
                p.grad = None



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

        if self.momentum_classifier is not None:
            outs = [out[2] for out in outs]

            log = {k: weighted_mean(outs, k, 'batch_size') for k in outs[0].keys() if k != 'batch_size'}

            self.log_dict(log, sync_dist=True)

