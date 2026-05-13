"""
高级论文级结果绘图
包含：柱状图、箱线图、雷达图、热图、森林图
符合顶级期刊标准 (Nature/Science/Lancet 等)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy import stats

# 设置专业绘图参数
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.0
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.dpi'] = 150

# 色盲友好配色方案
COLORS = {
    'coral': '#009E73',
    'ce': '#D55E00',
    'cdw_ce': '#F0E442',
    'mse': '#0072B2',
    'focal': '#E69F00',
    'ordinal': '#9467BD',
}

COLOR_BLIND = [
    '#009E73', '#D55E00', '#0072B2', '#F0E442', '#9467BD',
    '#E69F00', '#56B4E9', '#CC79A7'
]


class PaperFigures:
    """高级论文绘图类"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_performance_comparison(self, summary, figsize=(10, 6)):
        """
        性能对比柱状图 (带误差条和置信区间)
        适合: 主要结果展示
        """
        fig, ax = plt.subplots(figsize=figsize)

        loss_names = list(summary["overall"].keys())
        metrics = ['accuracy', 'qwk', 'mae']
        metric_labels = ['Accuracy', 'QWK', 'MAE']
        metric_units = ['%', '', '']

        x_pos = np.arange(len(loss_names))
        width = 0.25

        for i, (metric, label, unit) in enumerate(zip(metrics, metric_labels, metric_units)):
            means = [summary["overall"][ln][f"{metric}_mean"] for ln in loss_names]
            stds = [summary["overall"][ln][f"{metric}_std"] for ln in loss_names]
            cis_low = [summary["overall"][ln][f"{metric}_ci_low"] for ln in loss_names]
            cis_high = [summary["overall"][ln][f"{metric}_ci_high"] for ln in loss_names]

            offset = (i - 1) * width
            color = COLOR_BLIND[i]

            bars = ax.bar(x_pos + offset, means, width, yerr=stds,
                          capsize=3, label=label, color=color, alpha=0.8,
                          error_kw={'linewidth': 1.5})

            ax.errorbar(x_pos + offset, cis_low,
                       [means[j] - cis_low[j] for j in range(len(means))],
                       fmt='none', ecolor='black', alpha=0.3, linewidth=1)
            ax.errorbar(x_pos + offset, cis_high,
                       [cis_high[j] - means[j] for j in range(len(means))],
                       fmt='none', ecolor='black', alpha=0.3, linewidth=1)

            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + stds[j]*0.15,
                       f'{means[j]:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels([ln.upper() for ln in loss_names], fontsize=10)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'figure1_performance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"图1已保存: {save_path}")
        return save_path

    def plot_stability_analysis(self, summary, figsize=(10, 5)):
        """
        稳定性分析图
        适合: 展示模型对数据划分的敏感性
        """
        loss_names = list(summary["overall"].keys())

        within_stds = [summary["within_repeat"][ln]["fold_avg_std"] for ln in loss_names]
        between_stds = [summary["between_repeat"][ln]["accuracy_std"] for ln in loss_names]
        ratios = [b / (w + 1e-8) for b, w in zip(between_stds, within_stds)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 左图：标准差对比
        x_pos = np.arange(len(loss_names))
        width = 0.35

        bars1 = ax1.bar(x_pos - width/2, within_stds, width, label='Within-Fold',
                      color=COLOR_BLIND[0], alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, between_stds, width, label='Between-Repeat',
                      color=COLOR_BLIND[1], alpha=0.8)

        ax1.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([ln.upper() for ln in loss_names], fontsize=10, rotation=15)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax1.set_title('Variability Comparison', fontsize=11, fontweight='bold')

        # 右图：变异比
        colors = ['#28a745' if r < 0.8 else '#ffc107' if r < 1.5 else '#dc3545' for r in ratios]
        bars3 = ax2.bar(x_pos, ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        ax2.set_ylabel('Variation Ratio', fontsize=11, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([ln.upper() for ln in loss_names], fontsize=10, rotation=15)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax2.set_title('Sensitivity to Data Partition (Lower is Better)', fontsize=11, fontweight='bold')

        for bar, val in zip(bars3, ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'figure7_stability.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"稳定性分析图已保存: {save_path}")
        return save_path

    def plot_all(self, summary):
        """生成所有图表"""
        print(f"\n{'='*60}")
        print("生成高级论文图表...")
        print(f"{'='*60}")

        self.plot_performance_comparison(summary)
        self.plot_stability_analysis(summary)

        print(f"\n{'='*60}")
        print(f"所有图表已生成! 保存位置: {self.save_dir}")
        print(f"{'='*60}")

        return self.save_dir


# 使用示例
if __name__ == "__main__":
    pass
