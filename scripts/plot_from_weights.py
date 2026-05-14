"""
从已有权重加载模型并生成论文级图表
支持：多次五折交叉验证权重
"""
import os
import sys
import re
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

from config import Config
from core.dataset import FibrosisDataset
from core.model import MLPClassifier, CORALNet
from visualization.paper_figures import PaperFigures

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'liberation sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class WeightEvaluator:
    """从权重文件评估模型"""

    def __init__(self, weight_dir, config):
        self.weight_dir = weight_dir
        self.config = config
        self.loss_types = {
            'coral': 'coral',
            'ce': 'ce',
            'cdw_ce': 'cdw_ce',
            'cdw_ce_margin': 'cdw_ce_margin',
            'mse': 'mse'
        }

    def load_data(self):
        """加载完整数据集"""
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

    def parse_weight_filename(self, filename):
        """解析权重文件名
        例如: best_coral_r0_f2.pt -> loss='coral', repeat=0, fold=2
        """
        match = re.match(r'best_(\w+)_r(\d)_f(\d+)\.pt', filename)
        if match:
            return {
                'loss': match.group(1),
                'repeat': int(match.group(2)),
                'fold': int(match.group(3))
            }
        return None

    def load_model(self, weight_path, loss_type, input_dim, num_classes):
        """加载模型权重"""
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

        checkpoint = torch.load(weight_path, map_location=self.config.DEVICE)
        # 权重文件直接是state_dict，不是嵌套的
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(self.config.DEVICE)
        model.eval()

        return model

    def evaluate_model(self, model, X_test, y_test, loss_type):
        """评估模型"""
        test_dataset = FibrosisDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.config.DEVICE)

                if isinstance(model, CORALNet) or (hasattr(model, 'ordinal_head') and model.ordinal_head):
                    # CORAL: 输出 [B, K-1]
                    logits = model(batch_x)
                    probs = torch.sigmoid(logits)
                    preds = torch.sum(probs > 0.5, dim=1)
                else:
                    # 标准分类: 输出 [B, K]
                    logits = model(batch_x)
                    preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(batch_y)

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()

        # 计算指标
        accuracy = np.mean(preds == labels)
        adj_accuracy = np.mean(np.abs(preds - labels) <= 1)

        # QWK
        from sklearn.metrics import cohen_kappa_score
        qwk = cohen_kappa_score(labels, preds, weights='quadratic')

        # MAE
        mae = np.mean(np.abs(labels - preds))

        return {
            'accuracy': accuracy,
            'adjacent_accuracy': adj_accuracy,
            'qwk': qwk,
            'mae': mae,
            'preds': preds,
            'labels': labels
        }

    def run_evaluation(self):
        """运行所有权重评估"""
        print(f"\n{'='*80}")
        print(f"  从权重文件评估模型")
        print(f"{'='*80}")
        print(f"权重目录: {self.weight_dir}")

        # 加载数据
        X, y, le = self.load_data()
        print(f"总样本数: {len(X)}")
        print(f"标签分布: {np.bincount(y)}")

        input_dim = X.shape[1]
        num_classes = len(le.classes_)

        # 扫描权重文件
        weight_files = [f for f in os.listdir(self.weight_dir) if f.endswith('.pt')]
        print(f"找到 {len(weight_files)} 个权重文件")

        # 创建kfold划分 (使用与训练时相同的种子)
        seeds = [42, 123, 456]
        k = 5

        # 存储所有结果
        all_results = {
            'mlp_coral': [],  # MLP + CORAL loss
            'coral': [],      # CORALNet + CORAL loss
            'ce': [],
            'cdw_ce': [],
            'cdw_ce_margin': [],
            'mse': []
        }

        # 遍历每个repeat
        for repeat_idx, seed in enumerate(seeds):
            print(f"\n{'='*60}")
            print(f"  Repeat {repeat_idx + 1}/3 (seed={seed})")
            print(f"{'='*60}")

            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_test = X[test_idx]
                y_test = y[test_idx]

                # 标准化 (基于训练集)
                X_train = X[train_idx]
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # 评估每个loss
                for loss_name in all_results.keys():
                    weight_file = f"best_{loss_name}_r{repeat_idx}_f{fold_idx}.pt"
                    weight_path = os.path.join(self.weight_dir, weight_file)

                    if not os.path.exists(weight_path):
                        print(f"警告: 权重文件不存在: {weight_file}")
                        continue

                    # 加载模型
                    model = self.load_model(weight_path, loss_name, input_dim, num_classes)

                    # 评估
                    metrics = self.evaluate_model(model, X_test, y_test, loss_name)

                    print(f"{loss_name:<15} R{repeat_idx}F{fold_idx} | "
                          f"Acc: {metrics['accuracy']:.4f}, "
                          f"QWK: {metrics['qwk']:.4f}, "
                          f"MAE: {metrics['mae']:.4f}")

                    # 保存结果
                    result_entry = {
                        'repeat': repeat_idx,
                        'fold': fold_idx,
                        **metrics
                    }
                    all_results[loss_name].append(result_entry)

        # 汇总结果
        summary = self.compute_summary(all_results)

        return all_results, summary, le

    def compute_summary(self, all_results):
        """计算汇总统计"""
        summary = {
            "within_repeat": {},
            "between_repeat": {},
            "overall": {}
        }

        for loss_name, results in all_results.items():
            # 提取所有指标
            accuracies = [r['accuracy'] for r in results]
            qwks = [r['qwk'] for r in results]
            maes = [r['mae'] for r in results]

            # 计算统计量
            def compute_stats(values):
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                ci_low, ci_high = stats.t.interval(0.95, len(values)-1,
                                                   loc=mean, scale=std/np.sqrt(len(values)))
                return {
                    "mean": mean,
                    "std": std,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_samples": len(values)
                }

            summary["overall"][loss_name] = {
                "accuracy_mean": compute_stats(accuracies)["mean"],
                "accuracy_std": compute_stats(accuracies)["std"],
                "accuracy_ci_low": compute_stats(accuracies)["ci_low"],
                "accuracy_ci_high": compute_stats(accuracies)["ci_high"],
                "accuracy_n": compute_stats(accuracies)["n_samples"],
                "qwk_mean": compute_stats(qwks)["mean"],
                "qwk_std": compute_stats(qwks)["std"],
                "qwk_ci_low": compute_stats(qwks)["ci_low"],
                "qwk_ci_high": compute_stats(qwks)["ci_high"],
                "qwk_n": compute_stats(qwks)["n_samples"],
                "mae_mean": compute_stats(maes)["mean"],
                "mae_std": compute_stats(maes)["std"],
                "mae_ci_low": compute_stats(maes)["ci_low"],
                "mae_ci_high": compute_stats(maes)["ci_high"],
                "mae_n": compute_stats(maes)["n_samples"],
            }

            # 运行内统计
            repeat_means = []
            fold_stds = []
            for repeat in range(3):
                repeat_results = [r for r in results if r['repeat'] == repeat]
                if repeat_results:
                    repeat_accs = [r['accuracy'] for r in repeat_results]
                    repeat_means.append(np.mean(repeat_accs))
                    fold_stds.append(np.std(repeat_accs))

            summary["within_repeat"][loss_name] = {
                "fold_avg_std": np.mean(fold_stds) if fold_stds else 0
            }
            summary["between_repeat"][loss_name] = {
                "accuracy_std": np.std(repeat_means) if repeat_means else 0
            }

        return summary


def main():
    config = Config()

    # 权重目录
    weight_dir = r"/home/ubuntu/lq/MLP_results/20260513_054204/weights"

    # 输出目录
    output_dir = r"/home/ubuntu/lq/MLP_results/20260513_054204/figures"
    os.makedirs(output_dir, exist_ok=True)

    # 创建评估器
    evaluator = WeightEvaluator(weight_dir, config)

    # 运行评估
    all_results, summary, le = evaluator.run_evaluation()

    # 保存结果
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印最终汇总
    print(f"\n{'='*100}")
    print(f"  多次五折交叉验证最终结果")
    print(f"  数据: {len(le.classes_)}类 × {summary['overall'][list(summary['overall'].keys())[0]]['accuracy_n']}个评估点")
    print(f"{'='*100}")

    print(f"\n{'Loss':<12} {'Accuracy':>24} {'QWK':>24} {'MAE':>24}")
    print("-" * 90)

    for loss_name in ['coral', 'ce', 'cdw_ce', 'cdw_ce_margin', 'mse']:
        if loss_name in summary["overall"]:
            s = summary["overall"][loss_name]
            print(f"{loss_name:<12} {s['accuracy_mean']:>7.4f}±{s['accuracy_std']:<5.4f} "
                  f"[{s['accuracy_ci_low']:.4f},{s['accuracy_ci_high']:.4f}]  "
                  f"{s['qwk_mean']:>7.4f}±{s['qwk_std']:<5.4f} "
                  f"[{s['qwk_ci_low']:.4f},{s['qwk_ci_high']:.4f}]  "
                  f"{s['mae_mean']:>7.4f}±{s['mae_std']:<5.4f} "
                  f"[{s['mae_ci_low']:.4f},{s['mae_ci_high']:.4f}]")

    print(f"\n说明: 均值±标准差 [95%置信区间]")
    print(f"{'='*100}")

    # 生成论文级图表
    print(f"\n生成论文级图表...")
    plotter = PaperFigures(output_dir)
    plotter.plot_all(summary)

    print(f"\n所有图表已保存至: {output_dir}")


if __name__ == "__main__":
    main()
