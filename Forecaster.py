import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add project root
from decoder.mamba_decoder_lorentz import HyperbolicMambaDecoder
from Lifting.reconstructor import HyperbolicReconstructionHead
import pandas as pd
import torch
import torch.nn as nn
import geoopt

class HyperbolicSeqForecaster(nn.Module):
    """
    Inputs: manifold points from your existing encoders (trend_z, seasonal_z, resid_z),
            or directly a combined z0 if youve already fused them.
    Combines in tangent@origin, then autoregressively predicts H steps on the manifold,
    reconstructing each step to Euclidean outputs.
    """
    def __init__(self, embed_dim, hidden_dim, output_dim, manifold=None):
        super().__init__()
        self.manifold = manifold
        self.decoder = HyperbolicMambaDecoder(embed_dim, hidden_dim, self.manifold)
        self.recon   = HyperbolicReconstructionHead(embed_dim, output_dim, self.manifold)

    def combine_branches(self, *z_list):
        """
        z_list: iterable of manifold points [B, D+1]
        Combine by tangent sum at origin → expmap0.
        """
        u = 0
        for z in z_list:
            u = u + self.manifold.logmap0(z)
        z0 = self.manifold.projx(self.manifold.expmap0(u))
        return z0

    @torch.no_grad()
    def combine_only(self, *z_list):
        return self.combine_branches(*z_list)

    def forecast(self, pred_len, trend_z=None, seasonal_z=None, resid_z=None, z0=None,
                 teacher_forcing=False, z_true_seq=None, K=6):
        """
        Returns:
          x_hat:  [B, H, output_dim]  reconstructed Euclidean predictions
          z_pred: [B, H, D+1]         manifold predictions
        """
        assert (z0 is not None) or (trend_z is not None
         and seasonal_z is not None
          and resid_z is not None), \
            "Pass z0 or all of trend_z/seasonal_z/resid_z"

        if z0 is None:
            z_cur = self.combine_branches(trend_z, seasonal_z, resid_z)
        else:
            z_cur = z0

        preds_x = []
        preds_z = []
        for k in range(pred_len):
            z_next, _ = self.decoder(z_cur)
            x_hat_k = self.recon(z_next)
            preds_x.append(x_hat_k.unsqueeze(1))
            preds_z.append(z_next.unsqueeze(1))
            if teacher_forcing and z_true_seq is not None:
                z_cur = z_true_seq[:, k, :]
            else:
                z_cur = z_next
            if (k + 1) % K == 0:
                z_cur = z_cur.detach()

        x_hat = torch.cat(preds_x, dim=1)
        z_pred = torch.cat(preds_z, dim=1)
        return x_hat, z_pred
