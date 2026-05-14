# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch-based MLP classifier for liver fibrosis staging (F0–F4) from clinical biomarkers. The core goal is comparing multiple loss functions (CE, weighted CE, Focal, Label Smoothing) on an ordinal multi-class problem with severe class imbalance.

## Running the Code

```bash
# Full experiment: trains 4 loss variants, evaluates on test set, saves results + plots
python main.py

# Output goes to results/:
#   - best_{ce,weighted_ce,focal,label_smoothing}.pt   (model checkpoints)
#   - results.json                                      (metrics summary)
#   - metrics_comparison.png + training_curves.png      (plots)
```

## Architecture

### Data Pipeline (`dataset.py`)

- Source: Excel file with 391 rows × 25 columns (clinical indicators + `LABLE_F` label).
- Preprocessing order matters:
  1. **HA cleaning**: raw Excel contains malformed entries like `"19..56"`; `dataset.py` regex-cleans before `pd.to_numeric`.
  2. **Target encoding**: `LabelEncoder` maps `LABLE_F` (F0–F4) → integers 0–4.
  3. **Categorical one-hot**: `性别` and `Machine` are one-hot encoded; `编号`/`姓名` are dropped.
  4. **Standardization**: `StandardScaler` fit on train only; applied to val/test.
  5. **Stratified split**: 70% train / 10% val / 20% test.
  6. **Class weights**: computed as `1 / class_counts`, normalized to sum to num_classes. Used by weighted CE and Focal loss.

### Model (`model.py`)

Simple feed-forward MLP: `Linear → BatchNorm1d → ReLU → Dropout`, repeated per `HIDDEN_DIMS`, ending in `Linear(num_classes)`. Input dim is dynamic (depends on one-hot expansion).

### Loss Functions (`loss.py`)

| Name | Implementation | When to use |
|------|---------------|-------------|
| `ce` | `nn.CrossEntropyLoss` | Baseline |
| `weighted_ce` | `nn.CrossEntropyLoss(weight=class_weights)` | Class imbalance |
| `focal` | `FocalLoss(gamma=2.0, alpha=class_weights)` | Hard examples dominate |
| `label_smoothing` | `LabelSmoothingCrossEntropy(smoothing=0.1)` | Noisy labels |
| `ordinal` | `OrdinalCrossEntropy` (CORN-style) | **Not used in `main.py`**; requires model output dim = `num_classes - 1` |

`get_loss()` factory selects by string name.

### Training (`train.py`)

- `EarlyStopping` monitors **validation loss** (not accuracy).
- Best model state is restored after stopping.
- `train_model()` returns a history dict consumed by plotting in `utils.py`.

### Evaluation (`evaluate.py`)

In addition to standard metrics, `adjacent_accuracy` is computed as the proportion of predictions within ±1 grade of the ground truth. This is the clinically relevant metric for ordinal staging problems.

## Key Configuration (`config.py`)

All hyperparameters live in the `Config` class. Common changes:

- `DATA_PATH` / `OUTPUT_DIR`: Windows absolute paths; adjust if moving files.
- `HIDDEN_DIMS`, `DROPOUT`, `LEARNING_RATE`: model capacity and regularization.
- `EPOCHS` / `PATIENCE`: training length and early-stopping patience.
- `TEST_SIZE` / `VAL_SIZE`: data split ratios (must sum < 1).

To run a subset of losses, edit the `loss_configs` dict in `main.py`.

## Dependencies

```bash
pip install torch pandas numpy scikit-learn matplotlib openpyxl
```

## Notes

- The codebase assumes a Windows environment with Chinese paths. If cross-platform portability is needed, `DATA_PATH` and `OUTPUT_DIR` should be made relative or environment-driven.
- Batch size (32) with ~273 training samples yields ~9 batches per epoch. BatchNorm is stable at this size but may behave unexpectedly if batch size is reduced further.
- `OrdinalCrossEntropy` is implemented but **not wired into `main.py`** because the current `MLPClassifier` outputs `num_classes` logits, whereas ordinal loss expects `num_classes - 1`. To use it, change the model final layer output dimension and add `"ordinal"` to `loss_configs`.
