"""
Simple visualization script for training results.

Plots loss curves, sparsity metrics, and other key metrics.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def plot_training_history(history_path: str, save_path: str = None):
    """
    Plot training history from saved JSON file.
    
    Args:
        history_path: Path to history.json file
        save_path: Optional path to save figure
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_history = history['train']
    val_history = history['val']
    
    epochs = range(1, len(train_history) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Transcoder Training History', fontsize=16, fontweight='bold')
    
    # 1. Total Loss
    ax = axes[0, 0]
    train_loss = [h['total'] for h in train_history]
    ax.plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Reconstruction Loss
    ax = axes[0, 1]
    train_recon = [h['reconstruction'] for h in train_history]
    ax.plot(epochs, train_recon, 'g-', label='Train', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss (MSE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. L1 Sparsity
    ax = axes[0, 2]
    train_l1 = [h['l1_sparsity'] for h in train_history]
    ax.plot(epochs, train_l1, 'r-', label='Train', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L1 Sparsity')
    ax.set_title('L1 Sparsity Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Validation MSE
    ax = axes[1, 0]
    val_mse = [h['mse'] for h in val_history]
    ax.plot(epochs, val_mse, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Validation MSE')
    ax.grid(True, alpha=0.3)
    
    # 5. Validation Cosine Similarity
    ax = axes[1, 1]
    val_cos = [h['cosine_similarity'] for h in val_history]
    ax.plot(epochs, val_cos, 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Validation Cosine Similarity')
    ax.grid(True, alpha=0.3)
    
    # 6. Dead Features Fraction
    ax = axes[1, 2]
    val_dead = [h['dead_fraction'] for h in val_history]
    ax.plot(epochs, val_dead, 'r-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dead Fraction')
    ax.set_title('Validation Dead Features Fraction')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    
    # Print final metrics
    print("\n" + "="*60)
    print("Final Training Metrics:")
    print("="*60)
    final_train = train_history[-1]
    for k, v in final_train.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n" + "="*60)
    print("Final Validation Metrics:")
    print("="*60)
    final_val = val_history[-1]
    for k, v in final_val.items():
        print(f"  {k}: {v:.6f}")
    print("="*60)


def print_summary_stats(history_path: str):
    """Print summary statistics from training."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    val_history = history['val']
    
    # Get best epoch
    val_mse = [h['mse'] for h in val_history]
    best_epoch = np.argmin(val_mse) + 1
    best_metrics = val_history[best_epoch - 1]
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total epochs: {len(val_history)}")
    print(f"Best epoch: {best_epoch}")
    print(f"\nBest Validation Metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.6f}")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_training.py <path_to_history.json>")
        print("Example: python src/utils/visualize_training.py experiments/simple_run/history.json")
        sys.exit(1)
    
    history_path = sys.argv[1]
    
    if not Path(history_path).exists():
        print(f"Error: File not found: {history_path}")
        sys.exit(1)
    
    # Print summary
    print_summary_stats(history_path)
    
    # Plot
    save_path = Path(history_path).parent / "training_curves.png"
    plot_training_history(history_path, save_path=str(save_path))


