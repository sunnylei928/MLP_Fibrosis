"""
用已有5分类权重评估二分类任务 + 阈值优化分析
修复了colors变量作用域bug，新增了完整的阈值优化功能
"""
import os
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    roc_auc_score, f1_score, roc_curve
)

from config import Config
from core.dataset import FibrosisDataset
from core.model import MLPClassifier, CORALNet

# 设置绘图参数
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BinaryTaskEvaluator:
    """二分类任务评估器（含阈值优化）"""

    # 任务定义
    TASKS = {
        1: {"name": "F0 vs ≥F1", "threshold": 0, "desc": "检测任何纤维化",
            "pos_labels": [1, 2, 3, 4], "neg_labels": [0]},
        2: {"name": "≤F1 vs ≥F2", "threshold": 1, "desc": "显著纤维化",
            "pos_labels": [2, 3, 4], "neg_labels": [0, 1]},
        3: {"name": "≤F2 vs ≥F3", "threshold": 2, "desc": "进展期纤维化",
            "pos_labels": [3, 4], "neg_labels": [0, 1, 2]},
        4: {"name": "≤F3 vs F4", "threshold": 3, "desc": "肝硬化",
            "pos_labels": [4], "neg_labels": [0, 1, 2, 3]}
    }

    # 颜色配置（类级别，所有方法共享）
    COLORS = {
        'mlp_coral': '#56B4E9',     # MLP + CORAL (浅蓝)
        'coral': '#009E73',         # CORALNet + CORAL (青绿)
        'ce': '#D55E00',            # CE (橙)
        'cdw_ce': '#F0E442',        # CDW-CE (黄)
        'cdw_ce_margin': '#0072B2', # CDW-CE + Margin (蓝)
        'mse': '#9467BD'            # MSE (紫)
    }

    def __init__(self, weight_dir, config):
        self.weight_dir = weight_dir
        self.config = config
        self.device = config.DEVICE

    def load_data(self):
        """加载数据"""
        df = pd.read_excel(self.config.DATA_PATH)

        # 数据预处理
        df["HA"] = df["HA"].astype(str).str.replace(r"\.\..", ".", regex=True)
        df["HA"] = pd.to_numeric(df["HA"], errors="coerce")
        if df["HA"].isnull().sum() > 0:
            df["HA"] = df["HA"].fillna(df["HA"].median())

        # 编码标签
        le = LabelEncoder()
        df[self.config.TARGET_COL] = le.fit_transform(df[self.config.TARGET_COL])

        # 独热编码
        df = pd.get_dummies(df, columns=self.config.CATEGORICAL_COLS, drop_first=False)

        # 准备特征和标签
        feature_cols = [c for c in df.columns
                        if c not in self.config.DROP_COLS + [self.config.TARGET_COL]]
        X = df[feature_cols].values.astype(np.float32)
        y = df[self.config.TARGET_COL].values.astype(np.int64)

        return X, y, le

    def load_model(self, weight_path, loss_type, input_dim, num_classes):
        """加载模型"""
        # 根据损失类型选择架构
        if loss_type == 'coral':
            # CORALNet + CORAL loss (原始架构)
            model = CORALNet(input_dim, self.config.HIDDEN_DIMS, num_classes, self.config.DROPOUT)
        elif loss_type == 'mlp_coral':
            # MLP + CORAL loss (简化架构)
            model = MLPClassifier(input_dim, self.config.HIDDEN_DIMS, num_classes,
                                  self.config.DROPOUT, ordinal_head=True)
        else:
            # 标准分类损失
            model = MLPClassifier(input_dim, self.config.HIDDEN_DIMS, num_classes,
                                  self.config.DROPOUT, ordinal_head=False)

        checkpoint = torch.load(weight_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def predict_proba_5class(self, model, X_test):
        """获取5分类概率分布 [n_samples, n_classes]"""
        test_dataset = FibrosisDataset(X_test, np.zeros(len(X_test)))
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)

        all_probs = []

        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)

                if isinstance(model, CORALNet) or (hasattr(model, 'ordinal_head') and model.ordinal_head):
                    # CORAL: sigmoid输出 -> 转换为类别概率
                    logits = model(batch_x)  # [batch, num_classes-1]
                    prob_k = torch.sigmoid(logits)  # P(y > k)
                    # 转换为 P(y = k)
                    batch_probs = torch.zeros(batch_x.size(0), logits.size(1) + 1,
                                             device=self.device)
                    batch_probs[:, 0] = 1 - prob_k[:, 0]
                    for i in range(1, prob_k.size(1)):
                        batch_probs[:, i] = prob_k[:, i-1] - prob_k[:, i]
                    batch_probs[:, -1] = prob_k[:, -1]
                    # 归一化确保和为1
                    batch_probs = batch_probs / (batch_probs.sum(dim=1, keepdim=True) + 1e-8)
                else:
                    # 其他模型: logits -> softmax概率
                    logits = model(batch_x)  # [batch, num_classes]
                    batch_probs = F.softmax(logits, dim=1)

                all_probs.append(batch_probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)  # [n_samples, 5]

    def get_task_proba(self, probs_5class, task_id):
        """从5分类概率中提取指定二分类任务的阳性概率"""
        task_info = self.TASKS[task_id]
        pos_proba = np.sum(probs_5class[:, task_info["pos_labels"]], axis=1)
        return pos_proba

    def find_optimal_thresholds(self, y_true_binary, y_score):
        """
        阈值优化：扫描所有可能阈值，找到临床关键操作点
        返回: (optimal_dict, results_df)
        """
        if len(np.unique(y_true_binary)) < 2:
            return None, None

        thresholds = np.arange(0.0, 1.001, 0.001)
        results = []

        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)

            TP = np.sum((y_true_binary == 1) & (y_pred == 1))
            FP = np.sum((y_true_binary == 0) & (y_pred == 1))
            TN = np.sum((y_true_binary == 0) & (y_pred == 0))
            FN = np.sum((y_true_binary == 1) & (y_pred == 0))

            sens = TP / (TP + FN) if (TP + FN) > 0 else 0
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0
            f1 = 2*TP / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0
            youden = sens + spec - 1

            results.append({
                'threshold': thr,
                'sensitivity': sens,
                'specificity': spec,
                'f1': f1,
                'youden': youden
            })

        results_df = pd.DataFrame(results)

        # 找到各临床模式下的最优阈值
        optimal = {}

        # 1. Youden's J 最大（平衡模式）
        best_j_idx = results_df['youden'].idxmax()
        best_j = results_df.iloc[best_j_idx]
        optimal['balanced'] = {
            'threshold': float(best_j['threshold']),
            'sensitivity': float(best_j['sensitivity']),
            'specificity': float(best_j['specificity']),
            'f1': float(best_j['f1']),
            'youden': float(best_j['youden'])
        }

        # 2. F1 最优
        best_f1_idx = results_df['f1'].idxmax()
        best_f1 = results_df.iloc[best_f1_idx]
        optimal['f1_optimal'] = {
            'threshold': float(best_f1['threshold']),
            'sensitivity': float(best_f1['sensitivity']),
            'specificity': float(best_f1['specificity']),
            'f1': float(best_f1['f1']),
            'youden': float(best_f1['youden'])
        }

        # 3. 筛查模式: Sensitivity >= 0.90 且 Specificity 最高
        target_sens = 0.90
        screening_candidates = results_df[results_df['sensitivity'] >= target_sens]
        if len(screening_candidates) > 0:
            best_idx = screening_candidates['specificity'].idxmax()
            best = results_df.iloc[best_idx]
        else:
            best_idx = results_df['sensitivity'].idxmax()
            best = results_df.iloc[best_idx]
        optimal['screening'] = {
            'threshold': float(best['threshold']),
            'sensitivity': float(best['sensitivity']),
            'specificity': float(best['specificity']),
            'f1': float(best['f1']),
            'youden': float(best['youden'])
        }

        # 4. 确诊模式: Specificity >= 0.95 且 Sensitivity 最高
        target_spec = 0.95
        diagnosis_candidates = results_df[results_df['specificity'] >= target_spec]
        if len(diagnosis_candidates) > 0:
            best_idx = diagnosis_candidates['sensitivity'].idxmax()
            best = results_df.iloc[best_idx]
        else:
            best_idx = results_df['specificity'].idxmax()
            best = results_df.iloc[best_idx]
        optimal['diagnosis'] = {
            'threshold': float(best['threshold']),
            'sensitivity': float(best['sensitivity']),
            'specificity': float(best['specificity']),
            'f1': float(best['f1']),
            'youden': float(best['youden'])
        }

        # 默认阈值 0.5 的指标
        default_row = results_df[np.abs(results_df['threshold'] - 0.5) < 0.001]
        if len(default_row) > 0:
            d = default_row.iloc[0]
            optimal['default'] = {
                'threshold': 0.5,
                'sensitivity': float(d['sensitivity']),
                'specificity': float(d['specificity']),
                'f1': float(d['f1']),
                'youden': float(d['youden'])
            }

        return optimal, results_df

    def compute_binary_metrics(self, y_true, y_pred, y_proba=None):
        """计算二分类指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred, zero_division=0),
            'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['auc'] = 0.5

        return metrics

    def evaluate_model_on_binary_tasks(self, model, X_test, y_test, threshold_opt=True):
        """
        评估模型在4个二分类任务上的表现
        新增: threshold_opt=True 时进行阈值优化分析
        """
        # 获取5分类概率分布
        probs_5class = self.predict_proba_5class(model, X_test)

        results = {}
        threshold_results = {} if threshold_opt else None

        for task_id, task_info in self.TASKS.items():
            # 创建二分类标签
            binary_labels = (y_test > task_info['threshold']).astype(int)

            # 获取该任务的阳性概率
            pos_proba = self.get_task_proba(probs_5class, task_id)

            # 默认阈值 0.5 下的预测
            binary_preds_default = (pos_proba >= 0.5).astype(int)

            # 计算指标（使用概率计算AUC）
            metrics = self.compute_binary_metrics(binary_labels, binary_preds_default, pos_proba)
            results[f'task{task_id}'] = metrics

            # 阈值优化分析
            if threshold_opt:
                opt_result, all_thresholds = self.find_optimal_thresholds(
                    binary_labels, pos_proba
                )
                if opt_result:
                    threshold_results[f'task{task_id}'] = {
                        'optimal': opt_result,
                        'all_thresholds': all_thresholds,
                        'y_true': binary_labels,
                        'y_score': pos_proba
                    }

        return results, threshold_results

    def run_evaluation(self):
        """运行所有评估"""
        print(f"\n{'='*80}")
        print("  用5分类权重评估二分类任务")
        print("  （含阈值优化分析）")
        print(f"{'='*80}")
        print(f"权重目录: {self.weight_dir}")

        X, y, le = self.load_data()
        print(f"总样本数: {len(X)}")
        print(f"标签分布: {np.bincount(y)}")

        input_dim = X.shape[1]
        num_classes = len(le.classes_)

        weight_files = [f for f in os.listdir(self.weight_dir) if f.endswith('.pt')]
        print(f"找到 {len(weight_files)} 个权重文件")

        seeds = [42, 123, 456]
        k = 5

        all_results = {
            'coral': {'task1': [], 'task2': [], 'task3': [], 'task4': []},
            'ce': {'task1': [], 'task2': [], 'task3': [], 'task4': []},
            'cdw_ce': {'task1': [], 'task2': [], 'task3': [], 'task4': []},
            'cdw_ce_margin': {'task1': [], 'task2': [], 'task3': [], 'task4': []},
            'mse': {'task1': [], 'task2': [], 'task3': [], 'task4': []},
        }

        # 存储CORAL的阈值优化结果
        coral_threshold_results = []

        for repeat_idx, seed in enumerate(seeds):
            print(f"\n{'='*60}")
            print(f"  Repeat {repeat_idx + 1}/3 (seed={seed})")
            print(f"{'='*60}")

            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_test = X[test_idx]
                y_test = y[test_idx]

                X_train = X[train_idx]
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                for loss_name in all_results.keys():
                    weight_file = f"best_{loss_name}_r{repeat_idx}_f{fold_idx}.pt"
                    weight_path = os.path.join(self.weight_dir, weight_file)

                    if not os.path.exists(weight_path):
                        continue

                    model = self.load_model(weight_path, loss_name, input_dim, num_classes)

                    # 仅对CORAL进行阈值优化
                    do_thresh_opt = (loss_name == 'coral')
                    task_results, thresh_results = self.evaluate_model_on_binary_tasks(
                        model, X_test, y_test, threshold_opt=do_thresh_opt
                    )

                    for task_key, metrics in task_results.items():
                        all_results[loss_name][task_key].append(metrics)

                    # 保存CORAL的阈值优化结果
                    if do_thresh_opt and thresh_results:
                        coral_threshold_results.append({
                            'repeat': repeat_idx,
                            'fold': fold_idx,
                            'results': thresh_results
                        })

        summary = self.compute_summary(all_results)
        threshold_summary = self.aggregate_threshold_results(coral_threshold_results)

        return all_results, summary, threshold_summary

    def aggregate_threshold_results(self, coral_threshold_results):
        """聚合CORAL跨fold的阈值优化结果"""
        if not coral_threshold_results:
            return {}

        summary = {}

        for task_id in range(1, 5):
            task_key = f'task{task_id}'
            mode_stats = {}

            for mode in ['default', 'balanced', 'screening', 'diagnosis', 'f1_optimal']:
                values = []
                for entry in coral_threshold_results:
                    if task_key in entry['results']:
                        opt = entry['results'][task_key]['optimal']
                        if mode in opt:
                            values.append(opt[mode])

                if values:
                    sens_vals = [v['sensitivity'] for v in values]
                    spec_vals = [v['specificity'] for v in values]
                    thr_vals = [v['threshold'] for v in values]
                    f1_vals = [v['f1'] for v in values]
                    youden_vals = [v['youden'] for v in values]

                    mode_stats[mode] = {
                        'threshold_mean': float(np.mean(thr_vals)),
                        'threshold_std': float(np.std(thr_vals)) if len(values) > 1 else 0.0,
                        'sensitivity_mean': float(np.mean(sens_vals)),
                        'sensitivity_std': float(np.std(sens_vals)) if len(values) > 1 else 0.0,
                        'specificity_mean': float(np.mean(spec_vals)),
                        'specificity_std': float(np.std(spec_vals)) if len(values) > 1 else 0.0,
                        'f1_mean': float(np.mean(f1_vals)),
                        'youden_mean': float(np.mean(youden_vals)),
                    }

            summary[task_key] = mode_stats

        return summary

    def compute_summary(self, all_results):
        """计算汇总统计"""
        summary = {}

        for loss_name, task_results in all_results.items():
            summary[loss_name] = {}
            for task_id in ['task1', 'task2', 'task3', 'task4']:
                metrics_list = task_results[task_id]

                task_summary = {}
                for metric in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc']:
                    values = [m[metric] for m in metrics_list]
                    task_summary[f'{metric}_mean'] = float(np.mean(values))
                    task_summary[f'{metric}_std'] = float(np.std(values))

                summary[loss_name][task_id] = task_summary

        return summary

    def print_summary(self, summary):
        """打印汇总结果"""
        print(f"\n{'='*100}")
        print("  二分类任务评估结果汇总")
        print(f"{'='*100}")

        for task_id, task_info in self.TASKS.items():
            task_key = f'task{task_id}'
            print(f"\n{task_info['name']} - {task_info['desc']}")
            print(f"{'Loss':<15} {'AUC':>12} {'Sensitivity':>14} {'Specificity':>14} {'F1':>10}")
            print("-" * 70)

            for loss_name in ['mlp_coral', 'coral', 'ce', 'cdw_ce', 'cdw_ce_margin', 'mse']:
                if loss_name in summary and task_key in summary[loss_name]:
                    s = summary[loss_name][task_key]
                    auc_mean = s['auc_mean']
                    auc_std = s['auc_std']
                    sen_mean = s['sensitivity_mean']
                    sen_std = s['sensitivity_std']
                    spec_mean = s['specificity_mean']
                    spec_std = s['specificity_std']
                    f1_mean = s['f1_mean']

                    print(f"{loss_name:<15} {auc_mean:>6.4f}±{auc_std:<5.4f} "
                          f"{sen_mean:>6.4f}±{sen_std:<5.4f} "
                          f"{spec_mean:>6.4f}±{spec_std:<5.4f} "
                          f"{f1_mean:>8.4f}")

        print(f"\n{'='*100}")

    def print_threshold_summary(self, threshold_summary):
        """打印阈值优化结果汇总"""
        if not threshold_summary:
            return

        print(f"\n{'='*100}")
        print("  CORAL 阈值优化分析结果")
        print("  （从5分类概率提取阳性概率，扫描0-1阈值空间）")
        print(f"{'='*100}")

        mode_labels = {
            'default': '默认 (thr=0.5)',
            'balanced': "Youden's J 最优",
            'screening': '筛查模式 (Sens≥90%)',
            'diagnosis': '确诊模式 (Spec≥95%)',
            'f1_optimal': 'F1 最优'
        }

        for task_id, task_info in self.TASKS.items():
            task_key = f'task{task_id}'
            if task_key not in threshold_summary:
                continue

            print(f"\n{'─'*70}")
            print(f"  {task_info['name']} - {task_info['desc']}")
            print(f"{'─'*70}")
            print(f"{'Mode':<20} {'Threshold':>10} {'Sensitivity':>12} {'Specificity':>12} {'F1':>8}")
            print("-" * 66)

            for mode, label in mode_labels.items():
                if mode in threshold_summary[task_key]:
                    s = threshold_summary[task_key][mode]
                    print(f"{label:<20} "
                          f"{s['threshold_mean']:>6.3f}±{s['threshold_std']:<3.3f} "
                          f"{s['sensitivity_mean']:>6.3f}±{s['sensitivity_std']:<4.3f} "
                          f"{s['specificity_mean']:>6.3f}±{s['specificity_std']:<4.3f} "
                          f"{s['f1_mean']:>6.3f}")

    def plot_comparison(self, summary, save_dir):
        """绘制对比图"""
        os.makedirs(save_dir, exist_ok=True)

        losses = ['coral', 'ce', 'cdw_ce', 'cdw_ce_margin', 'mse']
        task_names = [f"Task {i}\n{self.TASKS[i]['name']}" for i in range(1, 5)]

        metrics_to_plot = ['auc', 'sensitivity', 'specificity', 'f1']
        metric_labels = ['AUC', 'Sensitivity', 'Specificity', 'F1 Score']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (metric, metric_label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[idx]

            data_matrix = np.zeros((len(losses), 4))
            std_matrix = np.zeros((len(losses), 4))

            for i, loss in enumerate(losses):
                for j in range(4):
                    task_key = f'task{j+1}'
                    if loss in summary and task_key in summary[loss]:
                        data_matrix[i, j] = summary[loss][task_key][f'{metric}_mean']
                        std_matrix[i, j] = summary[loss][task_key][f'{metric}_std']

            x = np.arange(4)
            width = 0.15

            for i, loss in enumerate(losses):
                offset = (i - 2) * width
                ax.bar(x + offset, data_matrix[i], width,
                      yerr=std_matrix[i], capsize=2,
                      label=loss.upper(), color=self.COLORS.get(loss, 'gray'),
                      alpha=0.8, error_kw={'linewidth': 1})

            ax.set_xlabel('Binary Classification Task', fontweight='bold')
            ax.set_ylabel(metric_label, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(task_names, fontsize=9)
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.set_ylim(0, 1.05)

        plt.suptitle('5-Class Models Evaluated on Binary Tasks',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'binary_task_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n对比图已保存: {save_path}")

        self.plot_heatmap(summary, save_dir)

    def plot_heatmap(self, summary, save_dir):
        """绘制热图对比（Bug修复：使用self.COLORS）"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        losses = ['coral', 'ce', 'cdw_ce', 'cdw_ce_margin', 'mse']

        for task_idx in range(4):
            task_key = f'task{task_idx+1}'
            ax = axes[task_idx]

            auc_data = []
            for loss in losses:
                if loss in summary and task_key in summary[loss]:
                    auc_data.append([summary[loss][task_key]['auc_mean']])

            colors_list = [self.COLORS.get(loss, 'gray') for loss in losses]
            ax.barh(losses, [d[0] for d in auc_data], color=colors_list, alpha=0.8)
            ax.set_xlim(0, 1)
            ax.set_title(f"{self.TASKS[task_idx+1]['name']}", fontweight='bold')
            ax.grid(True, axis='x', linestyle='--', alpha=0.3)

            for i, loss in enumerate(losses):
                if loss in summary and task_key in summary[loss]:
                    val = summary[loss][task_key]['auc_mean']
                    ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

        plt.suptitle('AUC Comparison Across Binary Tasks',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'binary_task_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"热图已保存: {save_path}")

    def plot_threshold_analysis(self, threshold_summary, save_dir):
        """绘制阈值优化分析图"""
        if not threshold_summary:
            print("\n无阈值优化数据，跳过绘图")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        mode_labels = {
            'default': 'Default (0.5)',
            'balanced': "Youden's J",
            'screening': 'Screening (Sens≥0.9)',
            'diagnosis': 'Diagnosis (Spec≥0.95)',
            'f1_optimal': 'F1 Optimal'
        }

        for task_idx in range(4):
            task_key = f'task{task_idx+1}'
            ax = axes[task_idx]

            if task_key not in threshold_summary:
                continue

            task_data = threshold_summary[task_key]
            modes = ['default', 'screening', 'diagnosis', 'balanced', 'f1_optimal']
            present_modes = [m for m in modes if m in task_data]

            x = np.arange(len(present_modes))
            sens_vals = [task_data[m]['sensitivity_mean'] for m in present_modes]
            sens_stds = [task_data[m].get('sensitivity_std', 0) for m in present_modes]
            spec_vals = [task_data[m]['specificity_mean'] for m in present_modes]
            spec_stds = [task_data[m].get('specificity_std', 0) for m in present_modes]

            width = 0.35
            ax.bar(x - width/2, sens_vals, width, yerr=sens_stds, capsize=3,
                  label='Sensitivity', color='#E63946', alpha=0.8,
                  error_kw={'linewidth': 1})
            ax.bar(x + width/2, spec_vals, width, yerr=spec_stds, capsize=3,
                  label='Specificity', color='#457B9D', alpha=0.8,
                  error_kw={'linewidth': 1})

            ax.set_title(f"{self.TASKS[task_idx+1]['name']}", fontweight='bold', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([mode_labels[m] for m in present_modes], rotation=30, ha='right', fontsize=8)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

            for i, m in enumerate(present_modes):
                thr = task_data[m]['threshold_mean']
                ax.text(i, 0.05, f"thr={thr:.2f}", ha='center', fontsize=7, color='navy')

        plt.suptitle('CORAL Threshold Optimization: Clinical Operating Points',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'threshold_optimization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"阈值优化图已保存: {save_path}")


def main():
    config = Config()

    weight_dir = r"/home/ubuntu/lq/MLP_results/5fold_3/weights"
    output_dir = r"/home/ubuntu/lq/MLP_results"
    os.makedirs(output_dir, exist_ok=True)

    evaluator = BinaryTaskEvaluator(weight_dir, config)

    all_results, summary, threshold_summary = evaluator.run_evaluation()

    evaluator.print_summary(summary)
    evaluator.print_threshold_summary(threshold_summary)

    with open(os.path.join(output_dir, 'binary_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if threshold_summary:
        with open(os.path.join(output_dir, 'threshold_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(threshold_summary, f, ensure_ascii=False, indent=2)

    evaluator.plot_comparison(summary, output_dir)
    evaluator.plot_threshold_analysis(threshold_summary, output_dir)

    print(f"\n所有结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()