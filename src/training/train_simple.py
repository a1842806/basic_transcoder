"""
Simple Transcoder Training Script

Train a vanilla transcoder (no Matryoshka nesting) to establish baseline.
"""

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import yaml
import os
import sys
from pathlib import Path
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.transcoder import Transcoder
from src.data.activation_collector import ActivationCollector, save_activations, load_activations
from src.training.losses import compute_transcoder_loss, compute_metrics, explained_variance


class SimpleTrainer:
    """Simple trainer for vanilla transcoder."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with config file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(self.config['device']['device'])
        self.dtype = getattr(torch, self.config['device']['dtype'])
        
        # Set seed for reproducibility
        torch.manual_seed(self.config['device']['seed'])
        
        # Create save directory
        self.save_dir = Path(self.config['logging']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.save_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"Initialized trainer. Saving to: {self.save_dir}")
        print(f"Device: {self.device}, dtype: {self.dtype}")
    
    def collect_or_load_activations(self):
        """Collect activations or load from disk if available."""
        act_path = self.config['data']['save_path']
        
        # Check if activations already exist
        if os.path.exists(act_path + '.npz'):
            print(f"Loading activations from {act_path}.npz...")
            pre_acts, post_acts = load_activations(act_path)
        else:
            print("Collecting activations...")
            collector = ActivationCollector(
                model_name=self.config['model']['name'],
                layer_idx=self.config['model']['layer_idx'],
                device=self.device,
                dtype=self.dtype
            )
            
            # Get sample texts (in real use, load from dataset like C4)
            texts = collector.get_sample_texts(n_samples=self.config['data']['n_texts'])
            
            # Collect activations
            pre_acts, post_acts = collector.collect_activations(
                texts=texts,
                max_tokens=self.config['data']['max_tokens'],
                batch_size=self.config['data']['batch_size'],
                collect_both=self.config['data']['collect_both']
            )
            
            # Save for future use
            os.makedirs(os.path.dirname(act_path), exist_ok=True)
            save_activations(pre_acts, post_acts, act_path)
        
        return pre_acts, post_acts
    
    def create_dataloaders(self, pre_acts, post_acts):
        """Create train and validation dataloaders."""
        # For transcoder: input is pre_acts, target is post_acts
        # If post_acts is None, we reconstruct pre_acts (like an autoencoder)
        if post_acts is None:
            print("Warning: No post-activations, training as autoencoder on pre-acts")
            post_acts = pre_acts
        
        # Create dataset
        dataset = TensorDataset(pre_acts, post_acts)
        
        # Split into train/val (90/10)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            drop_last=False
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def create_model_and_optimizer(self):
        """Initialize model and optimizer."""
        model = Transcoder(
            d_model=self.config['model']['d_model'],
            d_hidden=self.config['model']['d_hidden'],
            k=self.config['model']['k'],
            normalize_decoder=self.config['model']['normalize_decoder']
        ).to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model created with {n_params:,} parameters")
        
        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=self.config['training']['betas'],
            eps=self.config['training']['eps'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        return model, optimizer
    
    def train_epoch(self, model, optimizer, train_loader, epoch):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        total_metrics = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, (x, target) in enumerate(pbar):
            x = x.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            reconstruction, aux = model(x, return_aux=True)
            
            # Compute loss
            loss, loss_dict = compute_transcoder_loss(
                reconstruction=reconstruction,
                target=target,
                sparse_latents=aux['sparse_latents'],
                lambda_sparsity=self.config['training']['lambda_sparsity'],
                return_components=True
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Normalize decoder weights periodically
            if step % self.config['training']['normalize_decoder_every'] == 0:
                model.normalize_decoder_weights()
            
            # Update metrics
            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            
            # Update progress bar
            if step % self.config['logging']['log_every'] == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'l0': f"{aux['l0']:.1f}"
                })
        
        # Average metrics
        avg_metrics = {k: v / len(train_loader) for k, v in total_metrics.items()}
        return avg_metrics
    
    def evaluate(self, model, val_loader):
        """Evaluate on validation set."""
        model.eval()
        total_metrics = {}
        
        with torch.no_grad():
            for x, target in val_loader:
                x = x.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                reconstruction, aux = model(x, return_aux=True)
                
                # Compute metrics
                metrics = compute_metrics(
                    reconstruction=reconstruction,
                    target=target,
                    sparse_latents=aux['sparse_latents'],
                    active_mask=aux['active_mask']
                )
                
                # Compute R²
                r2 = explained_variance(reconstruction, target)
                metrics['r2'] = r2
                
                # Accumulate
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
        
        # Average metrics
        avg_metrics = {k: v / len(val_loader) for k, v in total_metrics.items()}
        return avg_metrics
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60 + "\n")
        
        # Step 1: Collect/load activations
        pre_acts, post_acts = self.collect_or_load_activations()
        
        # Step 2: Create dataloaders
        train_loader, val_loader = self.create_dataloaders(pre_acts, post_acts)
        
        # Step 3: Create model and optimizer
        model, optimizer = self.create_model_and_optimizer()
        
        # Step 4: Training loop
        best_val_loss = float('inf')
        history = {'train': [], 'val': []}
        
        for epoch in range(1, self.config['training']['n_epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['training']['n_epochs']}")
            
            # Train
            train_metrics = self.train_epoch(model, optimizer, train_loader, epoch)
            history['train'].append(train_metrics)
            
            # Evaluate
            val_metrics = self.evaluate(model, val_loader)
            history['val'].append(val_metrics)
            
            # Print metrics
            print(f"\nTrain Loss: {train_metrics['total']:.6f}")
            print(f"Val Metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.6f}")
            
            # Save best model
            if self.config['evaluation']['save_best'] and val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                }, self.save_dir / 'best_model.pt')
                print(f"✓ Saved best model (val MSE: {best_val_loss:.6f})")
        
        # Save final model
        torch.save({
            'epoch': self.config['training']['n_epochs'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, self.save_dir / 'final_model.pt')
        
        # Save training history
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Models saved to: {self.save_dir}")
        print("="*60 + "\n")
        
        return model, history


if __name__ == "__main__":
    # Get config path
    config_path = project_root / "config" / "simple_config.yaml"
    
    # Create trainer and train
    trainer = SimpleTrainer(str(config_path))
    model, history = trainer.train()
    
    print("✅ Training completed successfully!")


