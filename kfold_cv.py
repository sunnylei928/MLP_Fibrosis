"""
五折交叉验证实现方案
"""
import os
import numpy as np
import torch
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error

def run_kfold_cv(config, k=5):
    """
    五折交叉验证主函数

    Args:
        config: 配置对象
        k: 折数，默认 5

    Returns:
        all_results: 所有折的结果
        summary: 汇总统计
    """
    # 加载数据
    df = pd.read_excel(config.DATA_PATH)

    # 数据预处理
    df["HA"] = df["HA"].astype(str).str.replace(r"\.\.", ".", regex=True)
    df["HA"] = pd.to_numeric(df["HA"], errors="coerce")
    if df["HA"].isnull().sum() > 0:
        df["HA"] = df["HA"].fillna(df["HA"].median())

    # 编码标签
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[config.TARGET_COL] = le.fit_transform(df[config.TARGET_COL])

    # 独热编码分类特征
    df = pd.get_dummies(df, columns=config.CATEGORICAL_COLS, drop_first=False)

    # 准备特征和标签
    feature_cols = [c for c in df.columns
                    if c not in config.DROP_COLS + [config.TARGET_COL]]
    X = df[feature_cols].values.astype(np.float32)
    y = df[config.TARGET_COL].values.astype(np.int64)

    # 创建五折分层交叉验证
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_SEED)

    # 存储所有折的结果
    all_fold_results = {}

    # 遍历每个 loss 配置
    loss_configs = {
        "ce": ("ce", None),
        "cdw_ce": ("cdw_ce", None),
        "cdw_ce_margin": ("cdw_ce_margin", None),
        "mse": ("mse", None),
        "coral": ("coral", None),
    }

    for loss_name, (loss_type, weight) in loss_configs.items():
        print(f"\n{'='*60}")
        print(f"五折交叉验证: {loss_name}")
        print(f"{'='*60}")

        fold_results = []

        # 五折循环
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Fold {fold_idx + 1}/5 ---")

            # 划分数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 从训练集划分验证集 (10%)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, stratify=y_train, random_state=config.RANDOM_SEED
            )

            # 标准化（使用训练集统计）
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # 创建数据加载器
            train_loader = DataLoader(FibrosisDataset(X_train, y_train),
                                     batch_size=config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(FibrosisDataset(X_val, y_val),
                                   batch_size=config.BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(FibrosisDataset(X_test, y_test),
                                    batch_size=config.BATCH_SIZE, shuffle=False)

            input_dim = X_train.shape[1]
            num_classes = len(le.classes_)

            # 计算类别权重
            class_counts = np.bincount(y_train)
            class_weights = torch.FloatTensor(1.0 / (class_counts + 1e-6))
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = class_weights.to(config.DEVICE)

            # 创建模型
            if loss_type == 'coral':
                model = CORALNet(input_dim, config.HIDDEN_DIMS, num_classes, config.DROPOUT).to(config.DEVICE)
            else:
                model = MLPClassifier(input_dim, config.HIDDEN_DIMS, num_classes, config.DROPOUT).to(config.DEVICE)

            # 创建 loss
            loss_kwargs = {"num_classes": num_classes, "device": config.DEVICE}
            if loss_type == 'cdw_ce':
                loss_kwargs["alpha"] = 1.0
            elif loss_type == 'cdw_ce_margin':
                loss_kwargs["alpha"] = 1.0
                loss_kwargs["margin"] = 0.05

            criterion = get_loss(loss_type, class_weights=weight, **loss_kwargs)
            optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

            # 训练
            history = train_model(
                model, train_loader, val_loader, criterion, optimizer, config,
                loss_name=f"{loss_name}_fold{fold_idx}", save_dir=None  # 不保存
            )

            # 测试
            test_metrics = evaluate_epoch(model, test_loader, criterion, config.DEVICE)

            print(f"Fold {fold_idx + 1} - Acc: {test_metrics['accuracy']:.4f}, "
                  f"QWK: {test_metrics['qwk']:.4f}, MAE: {test_metrics['mae']:.4f}")

            fold_results.append(test_metrics)

        # 汇总该 loss 的五折结果
        all_fold_results[loss_name] = fold_results

    # 计算汇总统计
    summary = compute_summary(all_fold_results)

    return all_fold_results, summary


def compute_summary(all_fold_results):
    """
    计算五折结果的汇总统计

    Returns:
        summary: {
            "ce": {
                "accuracy_mean": 0.45,
                "accuracy_std": 0.02,
                "qwk_mean": 0.70,
                ...
            },
            ...
        }
    """
    summary = {}

    for loss_name, fold_results in all_fold_results.items():
        metrics_summary = {}

        # 对每个指标计算均值和标准差
        metric_names = ['accuracy', 'adjacent_accuracy', 'macro_f1', 'weighted_f1', 'qwk', 'mae']

        for metric in metric_names:
            values = [fold[metric] for fold in fold_results]
            metrics_summary[f"{metric}_mean"] = np.mean(values)
            metrics_summary[f"{metric}_std"] = np.std(values)

        summary[loss_name] = metrics_summary

    return summary


def print_kfold_summary(summary):
    """打印五折交叉验证汇总结果"""
    print(f"\n{'='*80}")
    print("五折交叉验证汇总结果 (均值 ± 标准差)")
    print(f"{'='*80}")

    print(f"\n{'Loss':<12} {'Accuracy':>20} {'QWK':>20} {'MAE':>20}")
    print("-" * 75)

    for loss_name, metrics in summary.items():
        acc_mean = metrics['accuracy_mean']
        acc_std = metrics['accuracy_std']
        qwk_mean = metrics['qwk_mean']
        qwk_std = metrics['qwk_std']
        mae_mean = metrics['mae_mean']
        mae_std = metrics['mae_std']

        print(f"{loss_name:<12} {acc_mean:>6.4f} ± {acc_std:<10.4f} "
              f"{qwk_mean:>6.4f} ± {qwk_std:<10.4f} "
              f"{mae_mean:>6.4f} ± {mae_std:<10.4f}")

    print(f"{'='*80}")


def save_kfold_results(all_fold_results, summary, save_dir):
    """保存五折交叉验证结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 保存汇总
    with open(os.path.join(save_dir, 'kfold_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存每折的详细结果
    with open(os.path.join(save_dir, 'kfold_details.json'), 'w', encoding='utf-8') as f:
        serializable = {}
        for loss_name, fold_results in all_fold_results.items():
            serializable[loss_name] = []
            for i, fold in enumerate(fold_results):
                fold_dict = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                             for k, v in fold.items() if k not in ['preds', 'labels']}
                fold_dict['fold'] = i
                serializable[loss_name].append(fold_dict)
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"五折结果已保存至: {save_dir}")
