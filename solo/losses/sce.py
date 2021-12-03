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

import torch
import torch.nn.functional as F


def sce_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    queue: torch.Tensor,
    temperature: float = 0.1,
    temperature_momentum: float = 0.05,
    lamb: float = 0.5,
) -> torch.Tensor:
    """Computes SCE's loss given a batch of queries from view 1, a batch of keys from view 2 and a
    queue of past elements.

    Args:
        query (torch.Tensor): NxD Tensor containing the queries from view 1.
        key (torch.Tensor): NxD Tensor containing the queries from view 2.
        queue (torch.Tensor): a queue of negative samples for the contrastive loss.
        temperature (float, optional): temperature of the softmax for z1. Defaults to 0.1.
        temperature_momentum (float, optional): temperature of the softmax for z2. Defaults to 0.05.
        lamb (float, optional): coefficient between contrastive and relational. Defaults to 0.5


    Returns:
        torch.Tensor: SCE loss.
    """

    b = z1.size(0)
    device = z1.device

    z2 = z2.detach()

    sim2_pos = torch.zeros(b, device=device)
    sim2_neg = torch.einsum("nc,kc->nk", [z2, queue])
    sim2 = torch.cat([sim2_pos.unsqueeze(1), sim2_neg], dim=1) / temperature_momentum
    p2 = F.softmax(sim2, dim=-1)
    w2 = lamb * F.one_hot(sim2_pos.to(int), p2.size(1)) + (1 - lamb) * p2

    sim1_pos = torch.einsum("nc,nc->n", z1, z2).unsqueeze(-1)
    sim1_neg = torch.einsum("nc,kc->nk", z1, queue)
    sim1 = torch.cat([sim1_pos, sim1_neg], dim=1) / temperature
    log_p1 = F.log_softmax(sim1, dim=-1)

    loss = -torch.sum(w2 * log_p1, dim=1).mean()

    return loss
