"""
Model Configuration Helper

Automatically detects model dimensions and creates appropriate configs
for different base models (GPT-2, Pythia, Gemma, etc.)
"""

import torch
from transformer_lens import HookedTransformer
from typing import Dict, Optional
import yaml
from pathlib import Path


def get_model_info(model_name: str, device: str = "cpu") -> Dict:
    """
    Load a model and extract its architectural information.
    
    Args:
        model_name: Name of the model (e.g., "gpt2", "pythia-160m", "gemma-2-2b")
        device: Device to load model on (use "cpu" to save memory)
    
    Returns:
        Dictionary with model information
    """
    print(f"Loading {model_name} to extract configuration...")
    
    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=torch.float32
        )
        
        info = {
            'model_name': model_name,
            'd_model': model.cfg.d_model,
            'n_layers': model.cfg.n_layers,
            'd_mlp': model.cfg.d_mlp if hasattr(model.cfg, 'd_mlp') else model.cfg.d_model * 4,
            'd_head': model.cfg.d_head,
            'n_heads': model.cfg.n_heads,
            'n_ctx': model.cfg.n_ctx,
            'act_fn': model.cfg.act_fn if hasattr(model.cfg, 'act_fn') else 'gelu',
        }
        
        # Recommended settings
        info['recommended'] = {
            'layer_idx': info['n_layers'] // 2,  # Middle layer
            'd_hidden': info['d_model'] * 4,  # 4x expansion (typical)
            'k': info['d_model'] // 4,  # ~25% sparsity
        }
        
        # Verify MLP hooks exist
        try:
            test_hook_pre = f"blocks.0.mlp.hook_pre"
            test_hook_post = f"blocks.0.mlp.hook_post"
            
            # Try to access hooks
            sample_text = "Test"
            tokens = model.to_tokens(sample_text)
            _, cache = model.run_with_cache(tokens, names_filter=[test_hook_pre, test_hook_post])
            
            if test_hook_pre in cache and test_hook_post in cache:
                info['mlp_hooks_valid'] = True
                info['hook_pre_name'] = "blocks.{layer_idx}.mlp.hook_pre"
                info['hook_post_name'] = "blocks.{layer_idx}.mlp.hook_post"
            else:
                info['mlp_hooks_valid'] = False
                print("⚠️ Warning: Standard MLP hooks not found. May need custom hook names.")
        except Exception as e:
            info['mlp_hooks_valid'] = False
            print(f"⚠️ Warning: Could not verify MLP hooks: {e}")
        
        print(f"✅ Model info extracted successfully")
        return info
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def print_model_info(info: Dict):
    """Pretty print model information."""
    print("\n" + "="*60)
    print(f"Model: {info['model_name']}")
    print("="*60)
    print(f"Architecture:")
    print(f"  d_model (hidden dim):     {info['d_model']}")
    print(f"  n_layers:                 {info['n_layers']}")
    print(f"  d_mlp:                    {info['d_mlp']}")
    print(f"  n_heads:                  {info['n_heads']}")
    print(f"  d_head:                   {info['d_head']}")
    print(f"  n_ctx (context length):   {info['n_ctx']}")
    print(f"  activation:               {info['act_fn']}")
    print(f"\nMLP Hooks:")
    print(f"  Valid: {info['mlp_hooks_valid']}")
    if info['mlp_hooks_valid']:
        print(f"  Pre:   {info['hook_pre_name']}")
        print(f"  Post:  {info['hook_post_name']}")
    print(f"\nRecommended Transcoder Settings:")
    print(f"  layer_idx:  {info['recommended']['layer_idx']} (middle layer)")
    print(f"  d_hidden:   {info['recommended']['d_hidden']} (4x expansion)")
    print(f"  k:          {info['recommended']['k']} (~25% sparsity)")
    print("="*60)


def create_config_for_model(
    model_name: str,
    config_name: str,
    base_config_path: str = "config/simple_config.yaml",
    output_dir: str = "config"
) -> str:
    """
    Create a config file for a specific model based on a template.
    
    Args:
        model_name: Name of the model (e.g., "gpt2")
        config_name: Name for the new config (e.g., "gpt2_config")
        base_config_path: Path to base config to use as template
        output_dir: Directory to save new config
    
    Returns:
        Path to created config file
    """
    # Get model info
    info = get_model_info(model_name, device="cpu")
    print_model_info(info)
    
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update model-specific settings
    config['model']['name'] = model_name
    config['model']['d_model'] = info['d_model']
    config['model']['layer_idx'] = info['recommended']['layer_idx']
    config['model']['d_hidden'] = info['recommended']['d_hidden']
    config['model']['k'] = info['recommended']['k']
    
    # Update save paths
    config['data']['save_path'] = f"experiments/{config_name}/activations"
    config['logging']['save_dir'] = f"experiments/{config_name}"
    config['logging']['wandb_run_name'] = config_name
    
    # Save new config
    output_path = Path(output_dir) / f"{config_name}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Created config: {output_path}")
    print(f"\nTo train with this config, run:")
    print(f"  python run_training.py {output_path}")
    
    return str(output_path)


def validate_config(config_path: str) -> bool:
    """
    Validate that a config file is compatible with its specified model.
    
    Args:
        config_path: Path to config file
    
    Returns:
        True if valid, False otherwise
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nValidating config: {config_path}")
    print("="*60)
    
    model_name = config['model']['name']
    info = get_model_info(model_name, device="cpu")
    
    issues = []
    
    # Check d_model matches
    if config['model']['d_model'] != info['d_model']:
        issues.append(
            f"d_model mismatch: config has {config['model']['d_model']}, "
            f"but {model_name} has {info['d_model']}"
        )
    
    # Check layer_idx is valid
    if config['model']['layer_idx'] >= info['n_layers']:
        issues.append(
            f"layer_idx {config['model']['layer_idx']} is out of bounds. "
            f"{model_name} only has {info['n_layers']} layers (0-{info['n_layers']-1})"
        )
    
    # Check k is reasonable
    if config['model']['k'] > config['model']['d_hidden']:
        issues.append(
            f"k ({config['model']['k']}) is larger than d_hidden ({config['model']['d_hidden']}). "
            f"This defeats the purpose of sparsity."
        )
    
    # Check MLP hooks
    if not info['mlp_hooks_valid']:
        issues.append(
            f"MLP hooks may not be valid for {model_name}. "
            f"Manual verification needed."
        )
    
    if issues:
        print("❌ Validation FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        print("="*60)
        return False
    else:
        print("✅ Config is valid!")
        print("="*60)
        return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/utils/model_config_helper.py info <model_name>")
        print("  python src/utils/model_config_helper.py create <model_name> <config_name>")
        print("  python src/utils/model_config_helper.py validate <config_path>")
        print("\nExamples:")
        print("  python src/utils/model_config_helper.py info gpt2")
        print("  python src/utils/model_config_helper.py create gpt2 gpt2_test")
        print("  python src/utils/model_config_helper.py validate config/simple_config.yaml")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "info":
        if len(sys.argv) < 3:
            print("Error: Please specify model name")
            print("Example: python src/utils/model_config_helper.py info gpt2")
            sys.exit(1)
        
        model_name = sys.argv[2]
        info = get_model_info(model_name)
        print_model_info(info)
    
    elif command == "create":
        if len(sys.argv) < 4:
            print("Error: Please specify model name and config name")
            print("Example: python src/utils/model_config_helper.py create gpt2 gpt2_test")
            sys.exit(1)
        
        model_name = sys.argv[2]
        config_name = sys.argv[3]
        create_config_for_model(model_name, config_name)
    
    elif command == "validate":
        if len(sys.argv) < 3:
            print("Error: Please specify config path")
            print("Example: python src/utils/model_config_helper.py validate config/simple_config.yaml")
            sys.exit(1)
        
        config_path = sys.argv[2]
        validate_config(config_path)
    
    else:
        print(f"Unknown command: {command}")
        print("Valid commands: info, create, validate")
        sys.exit(1)

