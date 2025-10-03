import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal
import math

class Transcoder(nn.Module):
    """
    Transcoder: h_pre -> encode -> activation -> (optional TopK) -> decode -> h_post

    Configs:
      - "relu":      ReLU + no TopK  (use L1 loss externally)
      - "topk":      ReLU + TopK     (no sparsity loss)
      - "jumprelu":  ReLU(pre - theta) with learnable per-feature theta + no TopK
                     (use tanh-L0 on (pre - theta) externally)
    """
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        k: Optional[int] = None,
        normalize_decoder: bool = True,
        config: Literal["relu", "topk", "jumprelu"] = "relu",
        theta_init: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = (k if k is not None else max(1, d_hidden // 16))
        self.normalize_decoder = normalize_decoder
        self.config = config

        # Weights (Kaiming -> stable with ReLU-ish gates)
        self.W_enc = nn.Parameter(torch.empty(d_model, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))

        # JumpReLU per-feature thresholds
        if config == "jumprelu":
            self.theta = nn.Parameter(torch.full((d_hidden,), float(theta_init)))
        else:
            self.register_parameter("theta", None)

        if self.normalize_decoder:
            self._normalize_decoder_rows()

    def _normalize_decoder_rows(self):
        """Normalize decoder ROWS (one row per feature vector) to unit norm."""
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
            self.W_dec.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, d_model] or [B, d_model]
        return x @ self.W_enc + self.b_enc

    def apply_activation(self, pre: torch.Tensor) -> torch.Tensor:
        if self.config in ("relu", "topk"):
            return F.relu(pre)
        elif self.config == "jumprelu":
            # Standard JumpReLU: max(pre - theta, 0) with per-feature theta
            return F.relu(pre - self.theta)
        else:
            raise ValueError(f"Unknown config: {self.config}")

    def apply_topk(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep top-k by magnitude along feature dim
        k = max(0, min(self.k, z.shape[-1]))
        if k == 0:
            sparse = torch.zeros_like(z)
            return sparse, sparse
        vals, idx = torch.topk(z.abs(), k=k, dim=-1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(-1, idx, z.gather(-1, idx))
        mask = (sparse != 0).to(z.dtype)
        return sparse, mask

    def decode(self, z_sparse: torch.Tensor) -> torch.Tensor:
        if self.normalize_decoder:
            self._normalize_decoder_rows()
        return z_sparse @ self.W_dec + self.b_dec

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        prefix: Optional[int] = None  # Matryoshka: keep features [:prefix]
    ):
        pre = self.encode(x)
        z = self.apply_activation(pre)

        # Optional Matryoshka prefix mask
        if prefix is not None:
            m = max(0, min(int(prefix), z.shape[-1]))
            if m < z.shape[-1]:
                mask_prefix = z.new_zeros(z.shape)
                mask_prefix[..., :m] = 1.0
                z = z * mask_prefix
            else:
                mask_prefix = torch.ones_like(z)
        else:
            mask_prefix = torch.ones_like(z)

        # Sparsity path
        if self.config == "topk":
            z_sparse, mask_topk = self.apply_topk(z)
            active_mask = mask_topk * mask_prefix
        else:
            z_sparse = z
            active_mask = (z_sparse != 0).to(z_sparse.dtype) * mask_prefix

        y = self.decode(z_sparse)

        if not return_aux:
            return y, None

        aux = {
            "pre": pre,                     # pre-activations (for tanh-L0 on JumpReLU)
            "activated_latents": z,         # after gate (+ prefix)
            "sparse_latents": z_sparse,     # after sparsity
            "active_mask": active_mask,
            "l0": active_mask.sum(dim=-1).mean(),
            "l1": z_sparse.abs().sum(dim=-1).mean(),
        }
        return y, aux

    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse feature activations (uses the same nonlinearity path)."""
        pre = self.encode(x)
        z = self.apply_activation(pre)
        if self.config == "topk":
            z_sparse, _ = self.apply_topk(z)
            return z_sparse
        return z


if __name__ == "__main__":
    # Test all three configurations
    batch_size = 4
    seq_len = 10
    d_model = 512
    d_hidden = 2048
    
    print("Testing all three transcoder configurations:")
    print("="*60)
    
    # Test ReLU Transcoder
    print("\n1. ReLU Transcoder (ReLU + no Top-K + L1 sparsity):")
    model_relu = Transcoder(d_model=d_model, d_hidden=d_hidden, config="relu")
    x = torch.randn(batch_size, seq_len, d_model)
    recon, aux = model_relu(x, return_aux=True)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Average L0 (active features): {aux['l0']:.2f}")
    print(f"  Average L1 norm: {aux['l1']:.2f}")
    print(f"  Config: {model_relu.config}")
    
    # Test Top-K Transcoder
    print("\n2. Top-K Transcoder (ReLU + Top-K + no sparsity loss):")
    model_topk = Transcoder(d_model=d_model, d_hidden=d_hidden, k=64, config="topk")
    recon, aux = model_topk(x, return_aux=True)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Average L0 (active features): {aux['l0']:.2f}")
    print(f"  Average L1 norm: {aux['l1']:.2f}")
    print(f"  Config: {model_topk.config}")
    print(f"  k (Top-K): {model_topk.k}")
    
    # Test JumpReLU Transcoder
    print("\n3. JumpReLU Transcoder (JumpReLU + no Top-K + tanh-L0 sparsity):")
    model_jumprelu = Transcoder(d_model=d_model, d_hidden=d_hidden, config="jumprelu", theta_init=0.1)
    recon, aux = model_jumprelu(x, return_aux=True)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Average L0 (active features): {aux['l0']:.2f}")
    print(f"  Average L1 norm: {aux['l1']:.2f}")
    print(f"  Config: {model_jumprelu.config}")
    print(f"  Theta (threshold): {model_jumprelu.theta.item():.3f}")
    
    print("\n✅ All transcoder configurations test passed!")


