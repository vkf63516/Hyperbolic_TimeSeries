import torch
import torch.nn as nn
import geoopt


class HorizonHyperbolicSegmentReconstructionHeadCD(nn.Module):
    """
    Channel-Dependent segment reconstructor for the multi-horizon forecaster.

    IMPORTANT — why this class exists instead of reusing
    HorizonHyperbolicSegmentReconstructionHead with output_dim=n_features:

    The original head's forward pass does:
        segment_flat.reshape(B * output_dim, N * segment_length)

    That reshape is only safe when output_dim == 1. For output_dim > 1 it
    flattens the whole [B, N, segment_length*output_dim] tensor and re-splits
    it along completely different axis boundaries, which interleaves
    segment-position and channel elements arbitrarily (verified: values from
    different segments and channels end up mixed within a single output row,
    and rows can even straddle batch elements depending on N/segment_length/
    output_dim). It silently produces garbage rather than erroring.

    This version instead does an explicit, shape-safe unflatten:
        [B, N, segment_length*output_dim]
          -> [B, N, segment_length, output_dim]   (unflatten last dim, safe)
          -> [B, N*segment_length, output_dim]    (merge adjacent dims, safe)

    which cleanly yields [B, pred_len, n_features].
    """

    def __init__(self, encode_dim, output_dim, num_pred_segments, segment_length, manifold, manifold_type,
                 hidden_dim=64, n_layers=1, dropout=0.1):
        super().__init__()
        self.manifold = manifold
        self.manifold_type = manifold_type
        self.num_pred_segments = num_pred_segments
        self.segment_length = segment_length
        self.output_dim = output_dim
        self.encode_dim = encode_dim

        layers = []

        # Input layer
        layers.append(nn.Linear(encode_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Output layer: segment_length * output_dim, ordered as
        # (segment_length, output_dim) — i.e. for a fixed time step within
        # the segment, all output_dim channel values are contiguous. This
        # ordering is what forward() below assumes when it unflattens.
        layers.append(nn.Linear(hidden_dim, segment_length * output_dim))

        self.fc = nn.Sequential(*layers)

    def forward(self, z_t):
        """
        z_t: [B, num_pred_segments, encode_dim] - point on hyperbolic manifold

        returns: [B, pred_len, output_dim]  (pred_len = num_pred_segments * segment_length)
        """
        B, N, D = z_t.shape
        v = self.manifold.logmap0(z_t)
        if self.manifold_type == "Lorentzian":
            if v.shape[-1] != self.encode_dim:
                # This can happen legitimately with Lorentz (input D+1, output D)
                print(f"?? Reconstructor: input dim {z_t.shape[-1]}, tangent dim {v.shape[-1]}, expected {self.encode_dim}")
            v = torch.clamp(v, min=-10.0, max=10.0)

        segment_flat = self.fc(v)  # [B, N, segment_length * output_dim]

        # Safe unflatten: split the last dim into (segment_length, output_dim)
        segment = segment_flat.view(B, N, self.segment_length, self.output_dim)

        # Merge segment-count and segment_length into the pred_len axis
        segment = segment.reshape(B, N * self.segment_length, self.output_dim)

        return segment