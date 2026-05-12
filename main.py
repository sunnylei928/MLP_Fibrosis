import os
import torch
import torch.optim as optim
from config import Config
from dataset import load_data
from model import MLPClassifier, CORALNet
from loss import get_loss
from train import train_model
from evaluate import evaluate_epoch, print_report
from utils import (save_results, plot_comparison, plot_confusion_matrices,
                   create_versioned_dir, save_run_metadata)

def main():
    config = Config()
    version_dir = create_versioned_dir(config.OUTPUT_DIR)
    print(f"\nVersion output dir: {version_dir}")

    # Load data
    train_loader, val_loader, test_loader, input_dim, num_classes, class_weights, le = load_data(config)
    print(f"Input dim: {input_dim}, Classes: {num_classes}")
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Loss functions to compare
    # 交叉熵 (ce)
    # 加权交叉熵 (weighted_ce)：结合了类别权重 (class_weights) 以缓解类别不平衡
    # Focal Loss (focal)
    # 标签平滑 (label_smoothing)
    # 类别距离加权交叉熵 (cdw_ce)：惩罚跨越多个等级的预测错误，权重参数 alpha=1.0
    # 带间隔的类别距离加权 (cdw_ce_margin)：在上述基础上增加了 margin=0.05
    # 均方误差 (mse)
    # CORAL Loss (coral)：搭配 CORALNet 模型使用
    loss_configs = {
        "ce":            ("ce", None),
        # "weighted_ce":   ("weighted_ce", class_weights),
        # "focal":         ("focal", class_weights),
        # "label_smoothing": ("label_smoothing", None),
        "cdw_ce":        ("cdw_ce", None),
        "cdw_ce_margin": ("cdw_ce_margin", None),
        "mse":           ("mse", None),
        "coral":         ("coral", None),
    }

    results = {}
    histories = {}

    for loss_name, (loss_type, weight) in loss_configs.items():
        print(f"\n{'='*60}")
        print(f"Training with Loss: {loss_name}")
        print(f"{'='*60}")

        # CORALNet requires a shared-weight head; others use standard MLP
        if loss_type == 'coral':
            model = CORALNet(input_dim, config.HIDDEN_DIMS, num_classes, config.DROPOUT).to(config.DEVICE)
        else:
            model = MLPClassifier(input_dim, config.HIDDEN_DIMS, num_classes, config.DROPOUT).to(config.DEVICE)

        loss_kwargs = {"num_classes": num_classes, "device": config.DEVICE}
        if loss_type == 'cdw_ce':
            loss_kwargs["alpha"] = 1.0
        elif loss_type == 'cdw_ce_margin':
            loss_kwargs["alpha"] = 1.0
            loss_kwargs["margin"] = 0.05

        criterion = get_loss(loss_type, class_weights=weight, **loss_kwargs)
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        history = train_model(
            model, train_loader, val_loader, criterion, optimizer, config,
            loss_name=loss_name, save_dir=version_dir
        )
        histories[loss_name] = history

        # Evaluate on test set
        test_metrics = evaluate_epoch(model, test_loader, criterion, config.DEVICE)
        print_report(test_metrics, le, title=f"Test ({loss_name})")
        results[loss_name] = test_metrics

    # Summary comparison
    print(f"\n{'='*60}")
    print("  Final Comparison (Test Set)")
    print(f"{'='*60}")
    # 扩展了打印宽度并加入了 QWK 和 MAE
    print(f"{'Loss':<16} {'Acc':>8} {'AdjAcc':>8} {'MacroF1':>8} {'QWK':>8} {'MAE':>8}")
    print("-" * 65)
    for name, m in results.items():
        print(f"{name:<16} {m['accuracy']:>8.4f} {m['adjacent_accuracy']:>8.4f} "
            f"{m['macro_f1']:>8.4f} {m['qwk']:>8.4f} {m['mae']:>8.4f}")
    print(f"{'='*60}")

    save_results(results, version_dir)
    plot_comparison(results, histories, version_dir)
    plot_confusion_matrices(results, le, version_dir)
    save_run_metadata(config, model, loss_configs, version_dir)
    print(f"\nResults and plots saved to: {version_dir}")

if __name__ == "__main__":
    main()
