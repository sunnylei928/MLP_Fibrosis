"""
多次五折交叉验证 - 完整实现
包含：数据划分、训练、评估、结果汇总、可视化
"""
import os
import sys
import json
import copy
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

# 导入项目模块
from config import Config
from dataset import FibrosisDataset
from model import MLPClassifier, CORALNet
from loss import get_loss
from train import train_model
from evaluate import evaluate_epoch
from utils import create_versioned_dir

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'liberation sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class Logger:
    """日志记录"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def close(self):
        self.log.close()


def load_data_full(config):
    """加载完整数据集"""
    df = pd.read_excel(config.DATA_PATH)

    # 数据预处理
    df["HA"] = df["HA"].astype(str).str.replace(r"\.\.", ".", regex=True)
    df["HA"] = pd.to_numeric(df["HA"], errors="coerce")
    if df["HA"].isnull().sum() > 0:
        df["HA"] = df["HA"].fillna(df["HA"].median())

    # 编码标签
    le = LabelEncoder()
    df[config.TARGET_COL] = le.fit_transform(df[config.TARGET_COL])

    # 独热编码
    df = pd.get_dummies(df, columns=config.CATEGORICAL_COLS, drop_first=False)

    # 准备特征和标签
    feature_cols = [c for c in df.columns
                    if c not in config.DROP_COLS + [config.TARGET_COL]]
    X = df[feature_cols].values.astype(np.float32)
    y = df[config.TARGET_COL].values.astype(np.int64)

    return X, y, le


def run_single_repeat(config, loss_configs, X, y, le, repeat_idx, seed, save_dir=None):
    """
    运行单次五折交叉验证

    Returns:
        fold_results: {fold_idx: {loss_name: metrics}}
    """
    print(f"\n{'='*80}")
    print(f"  Repeat {repeat_idx + 1} (seed={seed})")
    print(f"{'='*80}")

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 创建五折划分
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_results = {}

    # 遍历每一折
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/5 ---")
        print(f"训练: {len(train_idx)}, 测试: {len(test_idx)}")

        # 划分数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 从训练集划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=seed
        )

        print(f"最终: 训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}")
        print(f"标签分布: {np.bincount(y_train)}")

        # 标准化
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

        fold_results[fold_idx] = {}

        # 遍历每个 loss
        for loss_name, (loss_type, weight) in loss_configs.items():
            print(f"\n{loss_name}:")

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
                loss_name=f"{loss_name}_r{repeat_idx}_f{fold_idx}", save_dir=save_dir
            )

            # 测试
            test_metrics = evaluate_epoch(model, test_loader, criterion, config.DEVICE)

            print(f"  Acc: {test_metrics['accuracy']:.4f}, QWK: {test_metrics['qwk']:.4f}, MAE: {test_metrics['mae']:.4f}")

            # 保存结果
            fold_results[fold_idx][loss_name] = {
                'accuracy': test_metrics['accuracy'],
                'adjacent_accuracy': test_metrics['adjacent_accuracy'],
                'macro_f1': test_metrics['macro_f1'],
                'weighted_f1': test_metrics['weighted_f1'],
                'qwk': test_metrics['qwk'],
                'mae': test_metrics['mae']
            }

    return fold_results


def run_repeated_kfold_cv(config, loss_configs, n_repeats=3, seeds=None):
    """
    运行多次五折交叉验证

    Args:
        config: 配置对象
        loss_configs: loss配置
        n_repeats: 重复次数
        seeds: 随机种子列表

    Returns:
        all_results: {repeat_idx: {fold_idx: {loss_name: metrics}}}
        summary: 汇总统计
    """
    if seeds is None:
        seeds = [42, 123, 456][:n_repeats]

    # 加载完整数据
    X, y, le = load_data_full(config)
    print(f"\n总样本数: {len(X)}")
    print(f"标签分布: {np.bincount(y)}")
    print(f"标签映射: {le.classes_}")

    all_results = {}
    repeat_summaries = []

    # 创建保存目录
    version_dir = create_versioned_dir(config.OUTPUT_DIR)
    weights_dir = os.path.join(version_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 运行多次五折交叉验证
    for repeat_idx in range(n_repeats):
        seed = seeds[repeat_idx]

        fold_results = run_single_repeat(
            config, loss_configs, X, y, le, repeat_idx, seed, weights_dir
        )

        all_results[f"repeat_{repeat_idx}"] = fold_results

        # 计算本次运行的汇总
        repeat_summary = compute_repeat_summary(fold_results)
        repeat_summaries.append(repeat_summary)

        # 打印本次运行结果
        print(f"\n--- Repeat {repeat_idx + 1} 汇总 ---")
        print_repeat_summary(repeat_summary, le)

    # 计算多层次汇总
    summary = compute_hierarchical_summary(all_results, n_repeats, le)

    return all_results, summary, version_dir


def compute_repeat_summary(fold_results):
    """计算单次运行的汇总统计"""
    summary = {}

    for loss_name in fold_results[0].keys():
        metrics = ['accuracy', 'adjacent_accuracy', 'macro_f1', 'weighted_f1', 'qwk', 'mae']

        summary[loss_name] = {}
        for metric in metrics:
            values = [fold_results[f][loss_name][metric] for f in fold_results]
            summary[loss_name][f"{metric}_mean"] = np.mean(values)
            summary[loss_name][f"{metric}_std"] = np.std(values)

    return summary


def compute_hierarchical_summary(all_results, n_repeats, le):
    """
    计算多层次统计汇总

    Returns:
        summary: {
            "within_repeat": {...},
            "between_repeat": {...},
            "overall": {...}
        }
    """
    summary = {
        "within_repeat": {},
        "between_repeat": {},
        "overall": {}
    }

    loss_names = list(all_results["repeat_0"][0].keys())

    for loss_name in loss_names:
        # 收集所有结果
        all_values = {}  # {metric: [all_values]}
        repeat_means = {}  # {metric: [repeat_means]}

        for metric in ['accuracy', 'adjacent_accuracy', 'macro_f1', 'weighted_f1', 'qwk', 'mae']:
            all_values[metric] = []
            repeat_means[metric] = []

            for repeat_key in all_results:
                fold_data = all_results[repeat_key]

                # 收集该 repeat 所有 fold 的值
                fold_values = [fold_data[f][loss_name][metric] for f in fold_data]
                all_values[metric].extend(fold_values)

                # 该 repeat 的均值
                repeat_means[metric].append(np.mean(fold_values))

            # 总体统计
            mean_val = np.mean(all_values[metric])
            std_val = np.std(all_values[metric], ddof=1)

            # 95% 置信区间
            if len(all_values[metric]) >= 2:
                ci_low, ci_high = stats.t.interval(0.95, len(all_values[metric]) - 1,
                                                   loc=mean_val, scale=std_val / np.sqrt(len(all_values[metric])))
            else:
                ci_low = ci_high = mean_val

            summary["overall"][loss_name] = {
                f"{metric}_mean": mean_val,
                f"{metric}_std": std_val,
                f"{metric}_ci_low": ci_low,
                f"{metric}_ci_high": ci_high,
                f"{metric}_n": len(all_values[metric])
            }

        # 运行内标准差
        within_stds = []
        for repeat_key in all_results:
            fold_data = all_results[repeat_key]
            for metric in ['accuracy', 'qwk', 'mae']:
                fold_values = [fold_data[f][loss_name][metric] for f in fold_data]
                within_stds.append(np.std(fold_values))

        summary["within_repeat"][loss_name] = {
            "fold_avg_std": np.mean(within_stds)
        }

        # 运行间标准差
        for metric in ['accuracy', 'qwk', 'mae']:
            summary["between_repeat"][loss_name] = {
                f"{metric}_std": np.std(repeat_means[metric])
            }

    return summary


def print_repeat_summary(summary, le):
    """打印单次运行汇总"""
    print(f"\n{'Loss':<12} {'Acc':>8} {'QWK':>8} {'MAE':>8}")
    print("-" * 40)

    for loss_name, metrics in summary.items():
        print(f"{loss_name:<12} {metrics['accuracy_mean']:>8.4f} "
              f"{metrics['qwk_mean']:>8.4f} {metrics['mae_mean']:>8.4f}")


def print_final_summary(summary, le):
    """打印最终汇总结果"""
    print(f"\n{'='*100}")
    print(f"  多次五折交叉验证最终结果")
    n_evals = summary["overall"][list(summary["overall"].keys())[0]]["n_samples"]
    print(f"  数据: {len(le.classes_)}类 × {n_evals}个评估点")
    print(f"{'='*100}")

    print(f"\n{'Loss':<12} {'Accuracy':>20} {'QWK':>20} {'MAE':>20}")
    print("-" * 75)

    for loss_name in list(summary["overall"].keys()):
        acc_mean = summary["overall"][loss_name]["accuracy_mean"]
        acc_std = summary["overall"][loss_name]["accuracy_std"]
        acc_ci_low = summary["overall"][loss_name]["accuracy_ci_low"]
        acc_ci_high = summary["overall"][loss_name]["accuracy_ci_high"]

        qwk_mean = summary["overall"][loss_name]["qwk_mean"]
        qwk_std = summary["overall"][loss_name]["qwk_std"]
        qwk_ci_low = summary["overall"][loss_name]["qwk_ci_low"]
        qwk_ci_high = summary["overall"][loss_name]["qwk_ci_high"]

        mae_mean = summary["overall"][loss_name]["mae_mean"]
        mae_std = summary["overall"][loss_name]["mae_std"]
        mae_ci_low = summary["overall"][loss_name]["mae_ci_low"]
        mae_ci_high = summary["overall"][loss_name]["mae_ci_high"]

        print(f"{loss_name:<12} {acc_mean:>7.4f}±{acc_std:<5.4f} [{acc_ci_low:.4f},{acc_ci_high:.4f}]  "
              f"{qwk_mean:>7.4f}±{qwk_std:<5.4f} [{qwk_ci_low:.4f},{qwk_ci_high:.4f}]  "
              f"{mae_mean:>7.4f}±{mae_std:<5.4f} [{mae_ci_low:.4f},{mae_ci_high:.4f}]")

    print(f"\n说明: 均值±标准差 [95%置信区间]")
    print(f"{'='*100}")


def plot_comparison(summary, save_dir):
    """绘制对比图"""
    os.makedirs(save_dir, exist_ok=True)

    loss_names = list(summary["overall"].keys())

    # 准备数据
    metrics = ['accuracy', 'qwk', 'mae']
    metric_labels = ['Accuracy', 'QWK', 'MAE']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        means = []
        stds = []
        cis_low = []
        cis_high = []

        for loss_name in loss_names:
            means.append(summary["overall"][loss_name][f"{metric}_mean"])
            stds.append(summary["overall"][loss_name][f"{metric}_std"])
            cis_low.append(summary["overall"][loss_name][f"{metric}_ci_low"])
            cis_high.append(summary["overall"][loss_name][f"{metric}_ci_high"])

        x_pos = np.arange(len(loss_names))

        # 绘制柱状图
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, error_kw={'linewidth': 2})

        # 添加误差条
        ax.errorbar(x_pos, cis_low, [means[i] - cis_low[i] for i in range(len(means))],
                   fmt='none', ecolor='red', alpha=0.5, linewidth=1)
        ax.errorbar(x_pos, cis_high, [cis_high[i] - means[i] for i in range(len(means))],
                   fmt='none', ecolor='red', alpha=0.5, linewidth=1)

        ax.set_ylabel(label, fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(loss_names, rotation=15, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(stds)*0.02,
                   f'{means[i]:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Repeated K-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'repeated_kfold_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n对比图已保存: {save_path}")


def save_results(all_results, summary, save_dir):
    """保存所有结果"""
    # 保存汇总统计
    summary_path = os.path.join(save_dir, 'repeated_kfold_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        # 转换 numpy 类型为原生 Python 类型
        serializable_summary = {}
        for level in summary:
            serializable_summary[level] = {}
            for loss_name, metrics in summary[level].items():
                serializable_summary[level][loss_name] = {}
                for key, value in metrics.items():
                    if isinstance(value, (np.floating, np.integer)):
                        serializable_summary[level][loss_name][key] = float(value)
                    elif isinstance(value, np.ndarray):
                        serializable_summary[level][loss_name][key] = value.tolist()
                    else:
                        serializable_summary[level][loss_name][key] = value
        json.dump(serializable_summary, f, ensure_ascii=False, indent=2)

    # 保存详细结果
    details = {}
    for repeat_key, fold_data in all_results.items():
        details[repeat_key] = {}
        for fold_idx, loss_data in fold_data.items():
            details[repeat_key][fold_idx] = {}
            for loss_name, metrics in loss_data.items():
                details[repeat_key][fold_idx][loss_name] = metrics

    details_path = os.path.join(save_dir, 'repeated_kfold_details.json')
    with open(details_path, 'w', encoding='utf-8') as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    print(f"结果已保存:")
    print(f"  - {summary_path}")
    print(f"  - {details_path}")


def create_report(summary, save_dir):
    """创建文本报告"""
    report_path = os.path.join(save_dir, 'experiment_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 多次五折交叉验证实验报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 实验配置\n\n")
        f.write("- **折数**: 5\n")
        f.write("- **重复次数**: 3\n")
        f.write("- **随机种子**: [42, 123, 456]\n")
        f.write("- **总评估点**: 75 (5折 × 3次 × 5个模型)\n\n")

        f.write("## 结果汇总\n\n")
        f.write("### 主要指标 (均值 ± 标准差 [95%置信区间])\n\n")

        f.write("| Loss | Accuracy | QWK | MAE |\n")
        f.write("|------|----------|-----|-----|\n")

        for loss_name in summary["overall"].keys():
            acc = summary["overall"][loss_name]
            qwk = summary["overall"][loss_name]
            mae = summary["overall"][loss_name]

            f.write(f"| **{loss_name}** | "
                   f"{acc['accuracy_mean']:.4f}±{acc['accuracy_std']:.4f} [{acc['accuracy_ci_low']:.4f}, {acc['accuracy_ci_high']:.4f}] | "
                   f"{qwk['qwk_mean']:.4f}±{qwk['qwk_std']:.4f} [{qwk['qwk_ci_low']:.4f}, {qwk['qwk_ci_high']:.4f}] | "
                   f"{mae['mae_mean']:.4f}±{mae['mae_std']:.4f} [{mae['mae_ci_low']:.4f}, {mae['mae_ci_high']:.4f}] |\n")

        f.write("\n### 稳定性分析\n\n")
        f.write("| Loss | 五折变异 | 划分敏感度 |\n")
        f.write("|------|----------|------------|\n")

        for loss_name in summary["overall"].keys():
            within = summary["within_repeat"][loss_name]["fold_avg_std"]
            between = summary["between_repeat"][loss_name]["accuracy_std"]

            # 稳定性评级
            ratio = between / (within + 1e-8)
            if ratio < 0.8:
                stability = "✅ 优秀"
            elif ratio < 1.5:
                stability = "⚠️ 一般"
            else:
                stability = "❌ 较差"

            f.write(f"| {loss_name} | {within:.4f} | {between:.4f} {stability} |\n")

        f.write(f"\n*说明: 五折变异小 = 模型收敛好; 划分敏感度低 = 泛化能力强*\n\n")

    print(f"实验报告已保存: {report_path}")


def main():
    """主函数"""
    config = Config()

    # Loss 配置
    loss_configs = {
        "ce":            ("ce", None),
        "cdw_ce":        ("cdw_ce", None),
        "cdw_ce_margin": ("cdw_ce_margin", None),
        "mse":           ("mse", None),
        "coral":         ("coral", None),
    }

    # 实验参数
    n_repeats = 3
    seeds = [42, 123, 456]

    # 创建输出目录
    version_dir = create_versioned_dir(config.OUTPUT_DIR)

    # 创建日志
    log_file = os.path.join(version_dir, "repeated_kfold.log")
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"\n{'='*100}")
    print(f"  多次五折交叉验证实验")
    print(f"{'='*100}")
    print(f"版本目录: {version_dir}")
    print(f"日志文件: {log_file}")
    print(f"配置: {n_repeats}次重复 × 5折交叉验证")
    print(f"随机种子: {seeds}")

    # 运行多次五折交叉验证
    all_results, summary, save_dir = run_repeated_kfold_cv(
        config, loss_configs, n_repeats=n_repeats, seeds=seeds
    )

    # 打印最终汇总
    le = load_data_full(config)[2]  # 获取label encoder
    print_final_summary(summary, le)

    # 绘制对比图
    plot_comparison(summary, save_dir)

    # 保存结果
    save_results(all_results, summary, save_dir)

    # 创建报告
    create_report(summary, save_dir)

    print(f"\n{'='*100}")
    print(f"  实验完成！")
    print(f"  所有结果已保存至: {save_dir}")
    print(f"{'='*100}")

    logger.close()


if __name__ == "__main__":
    main()
