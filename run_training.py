#!/usr/bin/env python
"""
Convenient script to run transcoder training with different configs.

Usage:
    python run_training.py               # Uses simple_config.yaml
    python run_training.py quick_test    # Uses quick_test_config.yaml
    python run_training.py path/to/config.yaml  # Uses custom config
"""

import sys
from pathlib import Path
from src.training.train_simple import SimpleTrainer


def main():
    # Determine config path
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.endswith('.yaml'):
            config_path = Path(arg)
        else:
            # Assume it's a config name
            config_path = Path(f"config/{arg}_config.yaml")
    else:
        config_path = Path("config/simple_config.yaml")
    
    # Check if config exists
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("\nAvailable configs:")
        for cfg in Path("config").glob("*_config.yaml"):
            print(f"  - {cfg.stem.replace('_config', '')}")
        sys.exit(1)
    
    print(f"Using config: {config_path}")
    
    # Create trainer and run
    trainer = SimpleTrainer(str(config_path))
    model, history = trainer.train()
    
    print("\n✅ Training completed successfully!")
    print(f"Results saved to: {trainer.save_dir}")
    print(f"\nTo visualize results, run:")
    print(f"  python src/utils/visualize_training.py {trainer.save_dir}/history.json")


if __name__ == "__main__":
    main()


