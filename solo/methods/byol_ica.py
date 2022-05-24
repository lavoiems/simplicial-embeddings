import numpy as np
from solo.methods.byol import BYOL


class BYOL_ICA(BYOL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_ica = nn.ModuleList([nn.Linear(Z.shape[1], Z.shape[1], bias=False) for _ in range(5)])
        self.feats_ica = nn.ModuleList([nn.Linear(feats.shape[1], feats.shape[1], bias=False) for _ in range(5)])

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        out = super().forward(X, *args, **kwargs)
        for i in range(5):
            out[f'z_ica_{i}'] = self.z_ica[i](out['z'])
            out[f'feats_ica_{i}'] = self.feats_ica[i](out['feats'])
        return out

