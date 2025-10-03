"""
Loss functions for Transcoder training.

Includes:
- Reconstruction loss (MSE)
- Sparsity loss (L1)
- Combined loss with coefficients
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Mean Squared Error between reconstruction and target.
    
    Args:
        reconstruction: Reconstructed activations [batch, seq, d_model]
        target: Target activations [batch, seq, d_model]
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Reconstruction loss
    """
    return F.mse_loss(reconstruction, target, reduction=reduction)


def l1_sparsity_loss(
    sparse_latents: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    L1 penalty on sparse latent activations.
    
    Args:
        sparse_latents: Sparse latent activations [batch, seq, d_hidden]
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        L1 sparsity loss
    """
    l1 = sparse_latents.abs().sum(dim=-1)
    
    if reduction == "mean":
        return l1.mean()
    elif reduction == "sum":
        return l1.sum()
    else:
        return l1


def tanh_l0_sparsity_loss(
    sparse_latents: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Tanh-L0 sparsity loss: tanh(||z||_0) where ||z||_0 is the L0 norm.
    
    This smooths the L0 norm to make it differentiable.
    
    Args:
        sparse_latents: Sparse latent activations [batch, seq, d_hidden]
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Tanh-L0 sparsity loss
    """
    # Compute L0 norm (number of non-zero elements)
    l0_norm = (sparse_latents != 0).float().sum(dim=-1)
    
    # Apply tanh to smooth the L0 norm
    tanh_l0 = torch.tanh(l0_norm)
    
    if reduction == "mean":
        return tanh_l0.mean()
    elif reduction == "sum":
        return tanh_l0.sum()
    else:
        return tanh_l0


def compute_sparsity_loss(
    sparse_latents: torch.Tensor,
    sparsity_method: str,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute sparsity loss based on method.
    
    Args:
        sparse_latents: Sparse latent activations
        sparsity_method: 'l1', 'tanh_l0', or 'none'
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Sparsity loss
    """
    if sparsity_method == "l1":
        return l1_sparsity_loss(sparse_latents, reduction)
    elif sparsity_method == "tanh_l0":
        return tanh_l0_sparsity_loss(sparse_latents, reduction)
    elif sparsity_method == "none":
        return torch.tensor(0.0, device=sparse_latents.device)
    else:
        raise ValueError(f"Unknown sparsity method: {sparsity_method}")


def compute_transcoder_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    sparse_latents: torch.Tensor,
    lambda_sparsity: float = 1e-3,
    sparsity_method: str = "l1",
    return_components: bool = True
) -> tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    Combined loss for transcoder training.
    
    Args:
        reconstruction: Reconstructed activations
        target: Target activations
        sparse_latents: Sparse latent activations
        lambda_sparsity: Coefficient for sparsity loss
        sparsity_method: 'l1', 'tanh_l0', or 'none'
        return_components: Whether to return individual loss components
    
    Returns:
        Tuple of (total_loss, loss_dict if return_components else None)
    """
    # Reconstruction loss
    recon_loss = reconstruction_loss(reconstruction, target)
    
    # Sparsity loss
    sparsity_loss = compute_sparsity_loss(sparse_latents, sparsity_method)
    
    # Total loss
    total_loss = recon_loss + lambda_sparsity * sparsity_loss
    
    if return_components:
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
        }
        return total_loss, loss_dict
    
    return total_loss, None


def compute_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    sparse_latents: torch.Tensor,
    active_mask: torch.Tensor
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        reconstruction: Reconstructed activations
        target: Target activations
        sparse_latents: Sparse latent activations
        active_mask: Binary mask of active features
    
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # Reconstruction metrics
        mse = F.mse_loss(reconstruction, target).item()
        
        # Normalize for relative metrics
        target_norm = torch.norm(target, dim=-1, keepdim=True)
        error_norm = torch.norm(reconstruction - target, dim=-1, keepdim=True)
        relative_error = (error_norm / (target_norm + 1e-8)).mean().item()
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(reconstruction, target, dim=-1).mean().item()
        
        # Sparsity metrics
        l0 = active_mask.sum(dim=-1).mean().item()  # Average number of active features
        l1 = sparse_latents.abs().sum(dim=-1).mean().item()
        
        # Dead feature check (features that are never active in this batch)
        feature_activity = active_mask.sum(dim=(0, 1))  # Sum over batch and seq
        dead_features = (feature_activity == 0).sum().item()
        total_features = sparse_latents.shape[-1]
        dead_fraction = dead_features / total_features
        
        metrics = {
            'mse': mse,
            'relative_error': relative_error,
            'cosine_similarity': cos_sim,
            'l0': l0,
            'l1': l1,
            'dead_features': dead_features,
            'dead_fraction': dead_fraction,
        }
        
        return metrics


def explained_variance(
    reconstruction: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Compute explained variance (R²) of reconstruction.
    
    Args:
        reconstruction: Reconstructed activations
        target: Target activations
    
    Returns:
        Explained variance score (1.0 is perfect)
    """
    with torch.no_grad():
        # Variance of residuals
        residual_var = torch.var(target - reconstruction)
        
        # Variance of target
        target_var = torch.var(target)
        
        # R² = 1 - (residual_var / target_var)
        r2 = 1.0 - (residual_var / target_var).item()
        
        return r2


if __name__ == "__main__":
    # Test losses
    batch_size = 4
    seq_len = 10
    d_model = 512
    d_hidden = 2048
    k = 64
    
    # Create dummy data
    target = torch.randn(batch_size, seq_len, d_model)
    reconstruction = target + torch.randn_like(target) * 0.1  # Add small noise
    
    # Create sparse latents
    sparse_latents = torch.randn(batch_size, seq_len, d_hidden)
    topk_values, topk_indices = torch.topk(sparse_latents.abs(), k=k, dim=-1)
    mask = torch.zeros_like(sparse_latents)
    mask.scatter_(-1, topk_indices, 1.0)
    sparse_latents = sparse_latents * mask
    
    # Test different sparsity methods
    print("Testing L1 sparsity loss:")
    total_loss, loss_dict = compute_transcoder_loss(
        reconstruction, target, sparse_latents,
        lambda_sparsity=1e-3,
        sparsity_method="l1",
        return_components=True
    )
    
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    print("\nTesting Tanh-L0 sparsity loss:")
    total_loss, loss_dict = compute_transcoder_loss(
        reconstruction, target, sparse_latents,
        lambda_sparsity=1e-3,
        sparsity_method="tanh_l0",
        return_components=True
    )
    
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    print("\nTesting no sparsity loss:")
    total_loss, loss_dict = compute_transcoder_loss(
        reconstruction, target, sparse_latents,
        lambda_sparsity=1e-3,
        sparsity_method="none",
        return_components=True
    )
    
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    # Compute metrics
    active_mask = (sparse_latents != 0).float()
    metrics = compute_metrics(reconstruction, target, sparse_latents, active_mask)
    
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    
    r2 = explained_variance(reconstruction, target)
    print(f"\nExplained variance (R²): {r2:.6f}")
    
    print("\n✅ Loss functions test passed!")


