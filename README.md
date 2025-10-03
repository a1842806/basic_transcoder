# Matryoshka Transcoder Project - Phase 1: Simple Transcoder

**STATUS:** ✅ Phase 1 Complete - Ready for Training

## Project Structure

```
mtxplt/
├── config/                          # Configuration files
│   ├── simple_config.yaml           # Full training config (100k tokens)
│   └── quick_test_config.yaml       # Fast test config (5k tokens)
│
├── src/                             # Source code
│   ├── models/
│   │   └── transcoder.py            # Basic transcoder architecture
│   ├── data/
│   │   └── activation_collector.py  # Collect activations from Gemma-2-2B
│   ├── training/
│   │   ├── train_simple.py          # Training loop
│   │   └── losses.py                # Loss functions & metrics
│   ├── evaluation/
│   │   └── (empty - for Phase 2)
│   └── utils/
│       └── visualize_training.py    # Plot training curves
│
├── experiments/                     # Training outputs (auto-created)
├── notebooks/                       # For analysis (empty)
│
├── requirements.txt                 # Dependencies
├── test_components.py               # Component tests (✅ passing)
└── run_training.py                  # Convenient training launcher
```

## What We've Built (Phase 1)

### 1. ✅ Transcoder Architecture (`src/models/transcoder.py`)
- Vanilla transcoder: h_pre → encoder → TopK → decoder → h_post
- Configurable sparsity (TopK)
- Normalized decoder weights
- Returns auxiliary info (L0, L1, active features)
- Tested and working

### 2. ✅ Activation Collector (`src/data/activation_collector.py`)
- Uses TransformerLens to hook into Gemma-2-2B
- Collects MLP pre and post activations
- Supports any layer
- Saves/loads from disk
- Tested and working

### 3. ✅ Loss Functions (`src/training/losses.py`)
- Reconstruction loss (MSE)
- L1 sparsity loss
- Combined loss with coefficients
- Comprehensive metrics (MSE, R², cosine sim, L0, dead features)
- Tested and working

### 4. ✅ Training Pipeline (`src/training/train_simple.py`)
- Full training loop
- Train/val split
- Periodic decoder normalization
- Saves best model + final model
- Logs history to JSON
- Tested and working

### 5. ✅ Visualization (`src/utils/visualize_training.py`)
- Plots loss curves
- Shows sparsity metrics
- Prints summary statistics

## How to Use

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** You'll need:
- PyTorch with CUDA (for GPU)
- TransformerLens
- Hugging Face Transformers
- Access to Gemma-2-2B model weights

### Step 2: Test Components (Optional but Recommended)

```bash
python test_components.py
```

This verifies all components work independently before training.

### Step 3: Run Quick Test (5-10 minutes)

```bash
python run_training.py quick_test
```

This trains on just 5k tokens for 3 epochs to verify everything works end-to-end.

**Expected output:**
- Activations collected: 5000 tokens
- Training for 3 epochs
- Models saved to: `experiments/quick_test/`
- Final validation MSE should decrease over epochs

### Step 4: Run Full Training (1-2 hours)

```bash
python run_training.py simple
```

This trains on 100k tokens for 10 epochs (more realistic test).

**Expected output:**
- Activations collected: 100000 tokens
- Training for 10 epochs
- Models saved to: `experiments/simple_run/`
- Final validation metrics should show:
  - MSE < 0.01 (good reconstruction)
  - Cosine similarity > 0.95
  - L0 ≈ 64 (configured sparsity)
  - Dead fraction < 0.1 (< 10% dead features)

### Step 5: Visualize Results

```bash
python src/utils/visualize_training.py experiments/simple_run/history.json
```

This creates:
- Training curve plots
- Summary statistics
- Saved figure: `experiments/simple_run/training_curves.png`

### Step 6: Load Trained Model (for analysis)

```python
import torch
from src.models.transcoder import Transcoder

# Load best model
checkpoint = torch.load('experiments/simple_run/best_model.pt')
model = Transcoder(d_model=2304, d_hidden=9216, k=64)
model.load_state_dict(checkpoint['model_state_dict'])

# Use for inference
activations = ...  # Your activations
reconstructions = model(activations)
```

## Configurations

Two configs provided:

### 1. `quick_test_config.yaml` (FAST TESTING)
- 5k tokens, 3 epochs
- Smaller model (d_hidden=4608)
- Takes ~5-10 minutes
- Use for: debugging, sanity checks

### 2. `simple_config.yaml` (REAL TRAINING)
- 100k tokens, 10 epochs
- Full model (d_hidden=9216)
- Takes ~1-2 hours
- Use for: actual experiments

**Key parameters you might want to tune:**
- `model.d_hidden`: Latent dimension (4x to 8x of d_model typical)
- `model.k`: TopK sparsity (32-128 typical)
- `training.lambda_sparsity`: L1 coefficient (1e-4 to 1e-2)
- `training.learning_rate`: (1e-4 to 1e-3 typical)

## Expected Results (Sanity Checks)

### After quick_test:
- Loss should decrease (not increase)
- Validation MSE < 1.0
- L0 should match configured k (≈32 in quick test)
- Most features should be active (dead_fraction < 0.5)

### After simple training:
- Validation MSE < 0.01
- Cosine similarity > 0.95
- R² > 0.90 (explains >90% variance)
- Dead features < 10%
- L0 stable around configured k

### If results are worse:
- Check lambda_sparsity (too high? reduce to 1e-4)
- Check learning rate (too high? reduce to 1e-4)
- Check data quality (activations normalized?)

## Next Steps (Future Phases)

### Phase 2: Evaluation & Validation
- Load official Gemma transcoders for comparison
- Feature interpretability analysis
- Automated feature description
- Compare metrics side-by-side

### Phase 3: Matryoshka Architecture
- Extend Transcoder to support nested dictionaries
- Add auxiliary losses for nested levels
- Train with joint optimization

### Phase 4: Progressive Latent Training (PLT)
- Implement progressive training schedule
- Stage management
- Orthogonality constraints for new features
- Compare PLT vs joint training

### Phase 5: Full Evaluation
- Test hypothesis: reduced feature absorption
- Test hypothesis: reduced feature splitting
- Test hypothesis: fewer dead latents
- Circuit analysis using transcoders

### Phase 6: Optimization & Scale
- Multi-GPU training
- Streaming data loader
- FP16/BF16 training
- Scale to larger dictionaries (32k+)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce batch_size in config (try 128, 64, or 32) |
| TransformerLens can't load Gemma | Ensure you have HF access to Gemma-2-2B weights<br>Check: `huggingface-cli login` |
| Training loss not decreasing | Check lambda_sparsity (try 1e-4 instead of 1e-3)<br>Check learning rate (try 1e-4 instead of 1e-3) |
| All features dead | Reduce lambda_sparsity significantly (try 1e-5)<br>Increase learning rate (try 5e-3) |
| L0 too high (not sparse enough) | Increase lambda_sparsity (try 1e-2 or 5e-3)<br>Reduce k (TopK parameter) |

## Files Checklist

### Core (P0 - Required for training):
- ✅ `requirements.txt`
- ✅ `src/models/transcoder.py`
- ✅ `src/data/activation_collector.py`
- ✅ `src/training/losses.py`
- ✅ `src/training/train_simple.py`
- ✅ `config/simple_config.yaml`
- ✅ `config/quick_test_config.yaml`

### Utilities (P1 - Very useful):
- ✅ `test_components.py`
- ✅ `run_training.py`
- ✅ `src/utils/visualize_training.py`

### Future (P2 - Not yet implemented):
- ⏳ `src/models/matryoshka_transcoder.py`
- ⏳ `src/training/plt_trainer.py`
- ⏳ `src/evaluation/interpret.py`
- ⏳ `src/evaluation/compare_gemma.py`

## Testing Checklist

### Before running full training:
- ✅ Component tests pass (`python test_components.py`)
- ✅ Dependencies installed (`pip install -r requirements.txt`)
- ✅ GPU available (`torch.cuda.is_available()`)
- ✅ Gemma-2-2B accessible (`huggingface-cli login`)

### After quick test:
- [ ] Training completes without errors
- [ ] Loss decreases over epochs
- [ ] Models saved to `experiments/quick_test/`
- [ ] Can visualize results

### After full training:
- [ ] Validation MSE < 0.01
- [ ] Cosine similarity > 0.95
- [ ] Dead features < 10%
- [ ] Can load and use trained model

---

Good luck with training! 🚀

Questions or issues? Check the troubleshooting section or review component tests.

