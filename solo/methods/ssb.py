import torch
import torch.nn as nn
from torch.nn import functional as F


class ErasureChannel(nn.Module):
    def __init__(self, message_size, voc_size, channel_erase: float = 0.,
                 **kwargs):
        super().__init__()
        self.message_size = message_size
        self.voc_size = voc_size
        self.channel_erase = channel_erase

    def forward(self, x):
        B, L, V = x.shape
        erase = x.new_empty((B, L, 1)).bernoulli_(p=self.channel_erase)
        return torch.cat([(1 - erase) * x, erase], dim=-1)


class WSLinear(nn.Linear):
    def forward(self, x):
        dim = x.shape[-1]
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).detach()
        ws = weight - weight_mean
        weight_std = ws.pow(2).mean(dim=1, keepdim=True).sqrt().detach()
        ws = ws / (weight_std + 1e-8)
        return F.linear(x, ws[:, :dim], self.bias)


class SoftmaxBridge(nn.Module):
    def __init__(self, message_size, voc_size, emdim: int = 64, **kwargs):
        super().__init__()
        self.message_size = message_size
        self.voc_size = voc_size
        self.embedder = WSLinear(voc_size + 1, emdim)

    def forward(self, x,
                channel=None, tau=1., erase=None, gumbel=False, hard=False):
        tau = tau or self.tau
        channel = channel or (lambda x: x)
        erase = erase or (lambda x: x)

        ulogits = logits = x.view(-1, self.message_size, self.voc_size)
        logits = channel(logits)

        if gumbel and self.training:
            gumbels = torch.empty_like(logits).exponential_().log().neg()  # ~Gumbel(0,1)
            logits = logits + gumbels

        emb = F.softmax(logits / tau, dim=-1)
        if hard:
            top1 = F.one_hot(emb.argmax(-1), num_classes=emb.size(-1)).to(dtype=logits.dtype)
            emb = top1 - emb.detach() + emb

        emb = self.embedder(erase(emb))

        return ulogits, emb
