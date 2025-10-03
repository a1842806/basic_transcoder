"""
Activation Collector for Gemma-2-2B

Collects MLP pre-activations and post-activations from a specific layer
using TransformerLens hooks.
"""

import torch
from transformer_lens import HookedTransformer
from typing import Optional, List, Tuple
from tqdm import tqdm
import numpy as np


class ActivationCollector:
    """
    Collects activations from a transformer model's MLP layers.
    
    Args:
        model_name: Name of the model (e.g., "gemma-2-2b")
        layer_idx: Which layer to collect activations from
        device: Device to run model on
        dtype: Data type for model weights
    """
    
    def __init__(
        self,
        model_name: str = "gemma-2-2b",
        layer_idx: int = 12,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32
    ):
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.device = device
        self.dtype = dtype
        
        print(f"Loading {model_name} on {device}...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=dtype
        )
        self.model.eval()
        
        print(f"Model loaded! Collecting from layer {layer_idx}")
        print(f"Model has {self.model.cfg.n_layers} layers")
        print(f"MLP dimension: {self.model.cfg.d_model}")
    
    def collect_activations(
        self,
        texts: List[str],
        max_tokens: Optional[int] = None,
        batch_size: int = 8,
        collect_both: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Collect MLP activations from the specified layer.
        
        Args:
            texts: List of text strings to process
            max_tokens: Maximum number of tokens to collect (None = all)
            batch_size: Batch size for processing
            collect_both: If True, collect both pre and post activations
                         If False, only collect pre-activations
        
        Returns:
            Tuple of (pre_activations, post_activations)
            Each tensor has shape [n_tokens, d_model]
            post_activations is None if collect_both=False
        """
        pre_acts_list = []
        post_acts_list = [] if collect_both else None
        total_tokens = 0
        
        # Hook names - Use residual stream hooks for PLT transcoder
        # hook_mlp_in: MLP input at residual width (d_model)
        # hook_mlp_out: MLP output before residual add (d_model)
        hook_mlp_in = f"blocks.{self.layer_idx}.hook_mlp_in"
        hook_mlp_out = f"blocks.{self.layer_idx}.hook_mlp_out"
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                tokens = self.model.to_tokens(batch_texts)
                
                # Run with cache to collect activations
                if collect_both:
                    _, cache = self.model.run_with_cache(tokens, names_filter=[hook_mlp_in, hook_mlp_out])
                    pre_act = cache[hook_mlp_in]  # [batch, seq, d_model]
                    post_act = cache[hook_mlp_out]  # [batch, seq, d_model]
                else:
                    _, cache = self.model.run_with_cache(tokens, names_filter=[hook_mlp_in])
                    pre_act = cache[hook_mlp_in]  # [batch, seq, d_model]
                    post_act = None
                
                # CRITICAL: Validate dimensions match expected d_model
                expected_d_model = self.model.cfg.d_model
                assert pre_act.shape[-1] == expected_d_model, \
                    f"pre_act dimension mismatch: got {pre_act.shape[-1]}, expected {expected_d_model}"
                if post_act is not None:
                    assert post_act.shape[-1] == expected_d_model, \
                        f"post_act dimension mismatch: got {post_act.shape[-1]}, expected {expected_d_model}"
                
                # Flatten batch and seq dimensions: [batch * seq, d_model]
                pre_act_flat = pre_act.reshape(-1, pre_act.shape[-1])
                pre_acts_list.append(pre_act_flat.cpu())
                
                if collect_both:
                    post_act_flat = post_act.reshape(-1, post_act.shape[-1])
                    post_acts_list.append(post_act_flat.cpu())
                
                total_tokens += pre_act_flat.shape[0]
                
                # Check if we've collected enough
                if max_tokens is not None and total_tokens >= max_tokens:
                    break
        
        # Concatenate all activations
        pre_activations = torch.cat(pre_acts_list, dim=0)
        post_activations = torch.cat(post_acts_list, dim=0) if collect_both else None
        
        # Trim to max_tokens if specified
        if max_tokens is not None:
            pre_activations = pre_activations[:max_tokens]
            if post_activations is not None:
                post_activations = post_activations[:max_tokens]
        
        print(f"\nCollected {pre_activations.shape[0]} token activations")
        print(f"Activation shape: {pre_activations.shape}")
        print(f"Layer {self.layer_idx} | d_model={self.model.cfg.d_model}")
        print(f"Pre-activation stats:  mean={pre_activations.mean():.4f}, std={pre_activations.std():.4f}")
        if post_activations is not None:
            print(f"Post-activation stats: mean={post_activations.mean():.4f}, std={post_activations.std():.4f}")
        
        return pre_activations, post_activations
    
    def get_sample_texts(self, n_samples: int = 100) -> List[str]:
        """
        Get sample texts for quick testing.
        
        Args:
            n_samples: Number of sample sentences
        
        Returns:
            List of sample texts
        """
        # Simple diverse sentences for testing
        samples = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "The sun rises in the east and sets in the west.",
            "Climate change is one of the most pressing issues of our time.",
            "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
            "The Great Wall of China is visible from space.",
            "Quantum computers use qubits instead of classical bits.",
            "The human brain contains approximately 86 billion neurons.",
            "DNA carries genetic information in all living organisms.",
        ]
        
        # Repeat to get desired number
        full_samples = (samples * (n_samples // len(samples) + 1))[:n_samples]
        return full_samples


def save_activations(
    pre_acts: torch.Tensor,
    post_acts: Optional[torch.Tensor],
    save_path: str
):
    """
    Save activations to disk.
    
    Args:
        pre_acts: Pre-activations tensor
        post_acts: Post-activations tensor (optional)
        save_path: Path to save file (without extension)
    """
    save_dict = {'pre_activations': pre_acts.numpy()}
    if post_acts is not None:
        save_dict['post_activations'] = post_acts.numpy()
    
    np.savez_compressed(save_path + '.npz', **save_dict)
    print(f"Saved activations to {save_path}.npz")


def load_activations(load_path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load activations from disk.
    
    Args:
        load_path: Path to load file (without extension)
    
    Returns:
        Tuple of (pre_activations, post_activations)
    """
    data = np.load(load_path + '.npz')
    pre_acts = torch.from_numpy(data['pre_activations'])
    post_acts = torch.from_numpy(data['post_activations']) if 'post_activations' in data else None
    
    print(f"Loaded activations from {load_path}.npz")
    print(f"Shape: {pre_acts.shape}")
    
    return pre_acts, post_acts


if __name__ == "__main__":
    # Quick test with tiny dataset
    print("Testing Activation Collector...")
    
    collector = ActivationCollector(
        model_name="gemma-2-2b",
        layer_idx=12,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Get sample texts
    texts = collector.get_sample_texts(n_samples=20)
    
    # Collect activations
    pre_acts, post_acts = collector.collect_activations(
        texts=texts,
        max_tokens=1000,
        batch_size=4,
        collect_both=True
    )
    
    print(f"\n✅ Activation collection test passed!")
    print(f"Pre-activation stats: mean={pre_acts.mean():.3f}, std={pre_acts.std():.3f}")
    if post_acts is not None:
        print(f"Post-activation stats: mean={post_acts.mean():.3f}, std={post_acts.std():.3f}")


