# Simplicial Embeddings for Self-Supervised Learning and Downstream Classification
This repository is the companion code for the article Simplicial Embeddings for Self-supervised Learning and Downstream Classification. It is a fork of the Self-Supervised learning library `solo-learn` to which we apply the necessary modifications to run the experiments in the paper.

## SEM module
The SEM module can be implemented as follows:
```
class SEM(nn.Module):
    def __init__(self, L, V, tau, **kwargs):
        super().__init__()
        self.L = L
        self.V = V
        self.tau = tau

    def forward(self, x):
        logits = x.view(-1, self.L, self.V)
        taus = self.tau
        return F.softmax(logits / taus, -1).view(x.shape[0], -1)
```

## Citation
To cite our article, please cite:
```
@inproceedings{
    lavoie2023simplicial,
    title={Simplicial Embeddings in Self-Supervised Learning and Downstream Classification},
    author={Samuel Lavoie and Christos Tsirigotis and Max Schwarzer and Ankit Vani and Michael Noukhovitch and Kenji Kawaguchi and Aaron Courville},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=RWtGreRpovS}
}
```

To cite `solo-learn`, please cite their [paper](https://jmlr.org/papers/v23/21-1155.html):
```
@article{JMLR:v23:21-1155,
  author  = {Victor Guilherme Turrisi da Costa and Enrico Fini and Moin Nabi and Nicu Sebe and Elisa Ricci},
  title   = {solo-learn: A Library of Self-supervised Methods for Visual Representation Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {56},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v23/21-1155.html}
}
```
