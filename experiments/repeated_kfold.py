"""
多次五折交叉验证实现方案
评估模型对不同数据划分的稳定性
"""
import os
import sys
import json
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from scipy import stats

def run_repeated_kfold_cv(config, loss_configs, k=5, n_repeats=3):
    """
    多次五折交叉验证

    Args:
        config: 配置
        loss_configs: loss配置
        k: 折数
        n_repeats: 重复次数

    Returns:
        all_results: {repeat: {fold: {loss: metrics}}}
        summary: 多层次汇总统计
    """
    # 不同随机种子
    seeds = [42, 123, 456, 789, 1024][:n_repeats]

    # 存储所有结果
    all_results = {}

    for repeat_idx, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"  Repeat {repeat_idx + 1}/{n_repeats} (seed={seed})")
        print(f"{'='*80}")

        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 创建五折划分
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

        # 加载数据
        X, y, le = load_data_full(config)

        repeat_results = {fold_idx: {} for fold_idx in range(k)}

        # 遍历每一折
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Repeat {repeat_idx+1}, Fold {fold_idx+1}/5 ---")

            # 划分数据...
            # 训练和评估...
            # 保存结果到 repeat_results[fold_idx][loss_name]

        all_results[f"repeat_{repeat_idx}"] = repeat_results

    # 计算多层次汇总
    summary = compute_hierarchical_summary(all_results, k, n_repeats)

    return all_results, summary


def compute_hierarchical_summary(all_results, k, n_repeats):
    """
    计算多层次统计汇总

    Returns:
        {
            "within_repeat": {  # 单次运行内 (5折的变异)
                "coral": {"accuracy_std": 0.02, ...},
                ...
            },
            "between_repeat": {  # 多次运行间 (不同数据划分的变异)
                "coral": {"accuracy_std": 0.03, ...},
                ...
            },
            "overall": {  # 总体统计 (所有结果)
                "coral": {
                    "accuracy_mean": 0.50,
                    "accuracy_std": 0.025,
                    "accuracy_ci_low": 0.47,
                    "accuracy_ci_high": 0.53,
                    ...
                },
                ...
            }
        }
    """
    summary = {
        "within_repeat": {},
        "between_repeat": {},
        "overall": {}
    }

    # 收集所有结果
    for loss_name in loss_configs.keys():
        all_values = []  # 所有运行的所有折的结果

        repeat_means = []  # 每次运行的均值

        for repeat_key in all_results:
            repeat_data = all_results[repeat_key]
            fold_values = []

            for fold_idx in repeat_data:
                if loss_name in repeat_data[fold_idx]:
                    acc = repeat_data[fold_idx][loss_name]['accuracy']
                    fold_values.append(acc)
                    all_values.append(acc)

            if fold_values:
                repeat_means.append(np.mean(fold_values))

        if all_values:
            # 总体统计
            overall_mean = np.mean(all_values)
            overall_std = np.std(all_values, ddof=1)

            # 95% 置信区间
            ci_low, ci_high = stats.t.interval(0.95, len(all_values)-1,
                                               loc=overall_mean, scale=overall_std/np.sqrt(len(all_values)))

            summary["overall"][loss_name] = {
                "accuracy_mean": overall_mean,
                "accuracy_std": overall_std,
                "accuracy_ci_low": ci_low,
                "accuracy_ci_high": ci_high,
                "n_samples": len(all_values)
            }

            # 运行内标准差 (五折之间的变异)
            if repeat_means:
                summary["within_repeat"][loss_name] = {
                    "fold_std": np.mean([np.std([all_results[r][f][loss_name]['accuracy']
                                                for f in all_results[r] if loss_name in all_results[r][f]])
                                       for r in all_results])
                }

            # 运行间标准差 (不同数据划分之间的变异)
            summary["between_repeat"][loss_name] = {
                "repeat_std": np.std(repeat_means)
            }

    return summary


def print_repeated_kfold_summary(summary):
    """打印多次五折交叉验证汇总"""
    print(f"\n{'='*100}")
    print("  多次五折交叉验证汇总结果")
    print(f"{'='*100}")

    print(f"\n{'Loss':<10} {'Acc':>12} ± {'Std':>8} {'95% CI':>20} {'Within-Std':>12} {'Between-Std':>14}")
    print("-" * 100)

    for loss_name, stats in summary["overall"].items():
        mean = stats['accuracy_mean']
        std = stats['accuracy_std']
        ci_low = stats['accuracy_ci_low']
        ci_high = stats['accuracy_ci_high']
        within_std = summary["within_repeat"][loss_name]['fold_std']
        between_std = summary["between_repeat"][loss_name]['repeat_std']

        print(f"{loss_name:<10} {mean:>8.4f} ± {std:<6.4f} [{ci_low:.4f}, {ci_high:.4f}]  "
              f"{within_std:>10.4f}   {between_std:>12.4f}")

    print(f"\n说明:")
    print(f"  Acc ± Std: 总体均值和标准差")
    print(f"  95% CI: 95% 置信区间")
    print(f"  Within-Std: 五折之间的标准差 (模型稳定性)")
    print(f"  Between-Std: 不同数据划分之间的标准差 (对划分的敏感度)")
    print(f"{'='*100}")


def analyze_stability(summary):
    """
    分析模型稳定性

    Returns:
        recommendations: 建议和发现
    """
    print(f"\n{'='*100}")
    print("  稳定性分析")
    print(f"{'='*100}")

    recommendations = []

    for loss_name in summary["overall"].keys():
        between_std = summary["within_repeat"][loss_name]['repeat_std']
        within_std = summary["between_repeat"][loss_name]['fold_std']

        ratio = between_std / (within_std + 1e-8)

        if ratio > 2.0:
            recommendations.append({
                "loss": loss_name,
                "issue": "对数据划分敏感",
                "ratio": ratio,
                "suggestion": "需要更多数据或更正则化"
            })
        elif ratio < 0.5:
            recommendations.append({
                "loss": loss_name,
                "issue": "非常稳定",
                "ratio": ratio,
                "suggestion": "结果可靠，可推荐使用"
            })
        else:
            recommendations.append({
                "loss": loss_name,
                "issue": "稳定性一般",
                "ratio": ratio,
                "suggestion": "结果可接受"
            })

    print("\n模型稳定性评估:")
    print(f"{'Loss':<10} {'稳定性':>12} {'变异比':>10} {'建议':>30}")
    print("-" * 65)

    for rec in recommendations:
        print(f"{rec['loss']:<10} {rec['issue']:>12} {rec['ratio']:>10.2f} {rec['suggestion']:>30}")

    print(f"{'='*100}")

    return recommendations


# 配置示例
CONFIG = {
    "k": 5,              # 折数
    "n_repeats": 3,      # 重复次数
    "seeds": [42, 123, 456],  # 随机种子

    # 计算成本估算
    "estimated_time": {
        "single_fold": "5分钟",
        "single_kfold": "25分钟",  # 5 folds × 5 minutes
        "repeated_kfold": "75分钟"  # 3 repeats × 25 minutes
    },

    # 推荐配置
    "recommended": {
        "quick_test": {"k": 3, "n_repeats": 2, "losses": ["ce", "coral"]},
        "standard": {"k": 5, "n_repeats": 3, "losses": ["ce", "cdw_ce", "coral"]},
        "thorough": {"k": 5, "n_repeats": 5, "losses": ["ce", "cdw_ce", "mse", "coral"]}
    }
}
