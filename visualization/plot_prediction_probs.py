import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'liberation sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def get_prediction_probs(model, X, loss_name, num_classes=5):
    """
    获取模型预测的概率

    对于 MLPClassifier: 返回 softmax 概率 [B, num_classes]，即 P(y = k)
    对于 CORALNet: 返回累积概率 [B, num_classes-1]，即 P(y > k)
                   不转换为类别概率，直接用累积概率可视化
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)

        if 'coral' in loss_name.lower() or 'ordinal' in loss_name.lower():
            # CORAL/Ordinal: sigmoid 输出 P(y > k)，直接返回累积概率
            probs = torch.sigmoid(logits).cpu().numpy()  # [B, K-1]
        else:
            # CE, Focal, CDW-CE, MSE: softmax 输出 P(y = k)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # [B, K]

    return probs


def plot_single_loss_probs(loss_name, all_probs, all_preds, all_labels, label_names, save_dir):
    """
    为单个 loss 绘制 9 个随机样本的预测概率折线图

    CORAL: 绘制 4 个累积概率 P(y > k)
    其他: 绘制 5 个类别概率 P(y = k)
    """
    num_samples = min(9, len(all_probs))
    indices = np.random.choice(len(all_probs), num_samples, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    # 判断是否是 CORAL
    is_coral = 'coral' in loss_name.lower() or 'ordinal' in loss_name.lower()

    if is_coral:
        # CORAL: 4 个累积概率，X轴为 >F0, >F1, >F2, >F3
        # 累积概率应该是递降的：P(y>0) >= P(y>1) >= P(y>2) >= P(y>3)
        num_points = all_probs.shape[1]
        x_axis = np.arange(num_points)
        x_labels = [f'>{label_names[i]}' for i in range(num_points)]
        prob_type = 'Cumulative P(y > k)'
    else:
        # 其他: 5 个类别概率
        num_points = len(label_names)
        x_axis = np.arange(num_points)
        x_labels = label_names
        prob_type = 'Probability P(y = k)'

    for idx, sample_idx in enumerate(indices):
        ax = axes[idx]
        probs = all_probs[sample_idx]
        pred = all_preds[sample_idx]
        true_label = all_labels[sample_idx]

        # 绘制折线图
        ax.plot(x_axis, probs, marker='o', linewidth=2, markersize=8, color='steelblue')

        # 填充区域
        ax.fill_between(x_axis, probs, alpha=0.3, color='steelblue')

        # 标题颜色：预测正确绿色，错误红色
        title_color = 'green' if pred == true_label else 'red'
        ax.set_title(f'True: {label_names[true_label]} | Pred: {label_names[pred]}',
                     color=title_color, fontweight='bold', fontsize=11)

        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(prob_type, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

        # 在每个点上标注概率值
        for i, p in enumerate(probs):
            ax.text(i, p + 0.03, f'{p:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle(f'{loss_name.upper()} - {prob_type} (9 Random Samples)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'probs_{loss_name}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[{loss_name}] 概率折线图已保存: {save_path}")


def plot_all_losses_probs(models_dict, test_loader, device, label_names, save_dir, num_samples=9):
    """
    为所有 loss 模型生成预测概率折线图
    每个损失函数单独保存一张图

    Args:
        models_dict: {loss_name: (model, criterion)}
        test_loader: 测试数据加载器
        device: 设备
        label_names: 类别名称列表
        save_dir: 保存目录
        num_samples: 每个图显示的样本数
    """
    # 获取测试集所有数据
    all_X = []
    all_y = []
    for X, y in test_loader:
        all_X.append(X)
        all_y.append(y)
    all_X = torch.cat(all_X, dim=0).to(device)
    all_y = torch.cat(all_y, dim=0).cpu().numpy()

    # 为每个 loss 获取预测结果
    results = {}

    for loss_name, (model, criterion) in models_dict.items():
        # 获取概率
        probs = get_prediction_probs(model, all_X, loss_name)

        # 获取预测 - 统一使用 criterion.predict() 与 evaluate.py 保持一致
        logits = model(all_X)
        if hasattr(criterion, 'predict'):
            preds = criterion.predict(logits).cpu().numpy()
        else:
            preds = logits.argmax(dim=1).cpu().numpy()

        results[loss_name] = {
            'probs': probs,
            'preds': preds,
            'labels': all_y
        }

    # 为每个 loss 绘制单独的图
    for loss_name, data in results.items():
        plot_single_loss_probs(
            loss_name,
            data['probs'],
            data['preds'],
            data['labels'],
            label_names,
            save_dir
        )

    print(f"\n所有概率折线图已保存至: {save_dir}")


def plot_single_loss_probs_with_error(loss_name, all_probs, all_preds, all_labels,
                                       label_names, save_dir, show_errors_only=False):
    """
    为单个 loss 绘制预测概率折线图 - 可选择只显示预测错误的样本
    """
    if show_errors_only:
        # 只选择预测错误的样本
        error_indices = np.where(all_preds != all_labels)[0]
        if len(error_indices) == 0:
            print(f"[{loss_name}] 没有预测错误的样本！")
            return
        num_samples = min(9, len(error_indices))
        indices = error_indices[:num_samples]
        title_suffix = "(Error Samples Only)"
    else:
        num_samples = min(9, len(all_probs))
        indices = np.random.choice(len(all_probs), num_samples, replace=False)
        title_suffix = "(Random Samples)"

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    num_classes = len(label_names)
    x_axis = np.arange(num_classes)
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))

    for idx, sample_idx in enumerate(indices):
        if idx >= 9:
            break
        ax = axes[idx]
        probs = all_probs[sample_idx]
        pred = all_preds[sample_idx]
        true_label = all_labels[sample_idx]

        # 绘制折线图
        ax.plot(x_axis, probs, marker='o', linewidth=2, markersize=8, color='steelblue')

        # 填充区域
        ax.fill_between(x_axis, probs, alpha=0.3, color='steelblue')

        # 标题颜色：预测正确绿色，错误红色
        title_color = 'green' if pred == true_label else 'red'
        ax.set_title(f'True: {label_names[true_label]} | Pred: {label_names[pred]}',
                     color=title_color, fontweight='bold', fontsize=12)

        ax.set_xticks(x_axis)
        ax.set_xticklabels(label_names, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Probability', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

        # 在每个点上标注概率值
        for i, p in enumerate(probs):
            ax.text(i, p + 0.03, f'{p:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle(f'{loss_name.upper()} - Prediction Probabilities {title_suffix}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'probs_{loss_name}_errors.png' if show_errors_only else f'probs_{loss_name}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[{loss_name}] 概率折线图已保存: {save_path}")


def check_probability_smoothness(probs, pred_class, is_coral=False):
    """
    检查概率分布的平滑性/合理性

    对于 CORAL (累积概率): 计算违反单调性的次数和幅度
    对于其他 (类别概率): 计算"跨级预测倾向" - 远端类别概率是否过高
    """
    if is_coral:
        # CORAL: 累积概率应该单调递减
        # 统计上升的次数和幅度（违反单调性）
        violations = 0
        total_increase = 0.0
        for i in range(len(probs) - 1):
            if probs[i] < probs[i + 1]:
                violations += 1
                total_increase += probs[i + 1] - probs[i]
        return violations + total_increase * 10
    else:
        # 类别概率: 计算"跨级预测倾向"
        # CE 虽然单峰，但可能给远端类别分配较高概率
        cross_level_penalty = 0.0
        n = len(probs)

        for i in range(n):
            dist = abs(i - pred_class)
            if dist >= 2:  # 距离预测类别2级或更远
                # 例如预测F2，但F0或F4概率很高，这是不合理的
                cross_level_penalty += probs[i] * dist

        return cross_level_penalty * 5


def plot_ordinal_comparison(ordinal_model, ordinal_loss_name, standard_model, standard_loss_name,
                            test_loader, device, label_names, save_dir, num_samples=9):
    """
    对比有序回归和无序回归的概率分布

    CORAL: 显示 4 个累积概率 P(y > k)，应该单调递减
    Standard: 显示 5 个类别概率 P(y = k)
    """
    # 获取测试集所有数据
    all_X = []
    all_y = []
    for X, y in test_loader:
        all_X.append(X)
        all_y.append(y)
    all_X = torch.cat(all_X, dim=0).to(device)
    all_y = torch.cat(all_y, dim=0).cpu().numpy()

    # 获取两个模型的预测
    ordinal_probs = get_prediction_probs(ordinal_model, all_X, ordinal_loss_name)
    is_ordinal_coral = 'coral' in ordinal_loss_name.lower() or 'ordinal' in ordinal_loss_name.lower()

    if is_ordinal_coral:
        # CORAL: pred = sum(sigmoid(logits) > 0.5)
        logits = ordinal_model(all_X)
        ordinal_preds = (torch.sigmoid(logits) > 0.5).sum(dim=1).cpu().numpy()
    else:
        ordinal_preds = ordinal_probs.argmax(axis=1)

    standard_probs = get_prediction_probs(standard_model, all_X, standard_loss_name)
    standard_preds = standard_probs.argmax(axis=1)

    # 选择有代表性的样本：预测错误的、或概率分布不合理的
    interesting_samples = []

    for i in range(len(all_y)):
        # 优先选择预测错误的样本
        if ordinal_preds[i] != all_y[i] or standard_preds[i] != all_y[i]:
            interesting_samples.append(i)
        # 或者选择两个模型预测不一致的样本
        elif ordinal_preds[i] != standard_preds[i]:
            interesting_samples.append(i)

    if len(interesting_samples) < num_samples:
        # 补充随机样本
        remaining = set(range(len(all_y))) - set(interesting_samples)
        interesting_samples.extend(list(remaining)[:num_samples - len(interesting_samples)])

    np.random.shuffle(interesting_samples)
    selected_indices = interesting_samples[:num_samples]

    # 绘制对比图
    for idx, sample_idx in enumerate(selected_indices):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        ordinal_p = ordinal_probs[sample_idx]
        standard_p = standard_probs[sample_idx]
        true_label = all_y[sample_idx]

        # === 左图：有序回归 ===
        if is_ordinal_coral:
            # CORAL: 4 个累积概率
            x_axis_ord = np.arange(len(ordinal_p))
            x_labels_ord = [f'>{label_names[i]}' for i in range(len(ordinal_p))]
            prob_type_ord = 'Cumulative P(y > k)'
            color_ord = '#2ca02c'
        else:
            # 其他有序回归: 5 个类别概率
            x_axis_ord = np.arange(len(label_names))
            x_labels_ord = label_names
            prob_type_ord = 'Probability P(y = k)'
            color_ord = '#2ca02c'

        ax1.plot(x_axis_ord, ordinal_p, marker='o', linewidth=2.5, markersize=8,
                color=color_ord, label=ordinal_loss_name.upper())
        ax1.fill_between(x_axis_ord, ordinal_p, alpha=0.3, color=color_ord)

        # 检查平滑性
        smoothness = check_probability_smoothness(ordinal_p, ordinal_preds[sample_idx], is_coral=is_ordinal_coral)

        title_color = 'green' if ordinal_preds[sample_idx] == true_label else 'red'
        ax1.set_title(f'{ordinal_loss_name.upper()}\nTrue: {label_names[true_label]} | Pred: {label_names[ordinal_preds[sample_idx]]}',
                     color=title_color, fontweight='bold', fontsize=11)
        ax1.set_xticks(x_axis_ord)
        ax1.set_xticklabels(x_labels_ord)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_ylabel(prob_type_ord, fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right')

        # 标注概率值
        for i, p in enumerate(ordinal_p):
            ax1.text(i, p + 0.03, f'{p:.2f}', ha='center', va='bottom', fontsize=8)

        # === 右图：标准回归 ===
        x_axis_std = np.arange(len(standard_p))
        ax2.plot(x_axis_std, standard_p, marker='o', linewidth=2.5, markersize=8,
                color='#d62728', label=standard_loss_name.upper())
        ax2.fill_between(x_axis_std, standard_p, alpha=0.3, color='#d62728')

        # 检查平滑性
        smoothness_std = check_probability_smoothness(standard_p, standard_preds[sample_idx], is_coral=False)

        title_color_std = 'green' if standard_preds[sample_idx] == true_label else 'red'
        ax2.set_title(f'{standard_loss_name.upper()}\nTrue: {label_names[true_label]} | Pred: {label_names[standard_preds[sample_idx]]}',
                     color=title_color_std, fontweight='bold', fontsize=11)
        ax2.set_xticks(x_axis_std)
        ax2.set_xticklabels(label_names)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel('Probability P(y = k)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')

        # 标注概率值
        for i, p in enumerate(standard_p):
            ax2.text(i, p + 0.03, f'{p:.2f}', ha='center', va='bottom', fontsize=8)

        # 添加说明
        if is_ordinal_coral:
            violation_text_ord = "✓ Monotonic" if smoothness < 0.1 else f"Increases: {smoothness:.2f}"
        else:
            violation_text_ord = "✓ Smooth" if smoothness < 0.1 else f"Irregularity: {smoothness:.2f}"
        violation_text_std = "✓ Smooth" if smoothness_std < 0.1 else f"Irregularity: {smoothness_std:.2f}"

        fig.text(0.5, 0.02, f'Smoothness: {violation_text_ord} | {violation_text_std}',
                ha='center', fontsize=10, style='italic')

        plt.suptitle(f'Sample {idx + 1}: Ordinal vs Standard Regression Comparison',
                    fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'ordinal_vs_standard_{idx + 1}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"\n[对比] 有序 vs 标准 回归对比图已保存至: {save_dir}")
    print(f"      共生成 {len(selected_indices)} 张对比图")


def plot_smoothness_statistics(models_dict, test_loader, device, label_names, save_dir):
    """
    统计并可视化各模型的关键指标

    CORAL: 累积概率单调性违反
    其他: 跨级预测倾向（远端类别概率过高）
    """
    # 获取测试集所有数据
    all_X = []
    all_y = []
    for X, y in test_loader:
        all_X.append(X)
        all_y.append(y)
    all_X = torch.cat(all_X, dim=0).to(device)
    all_y = torch.cat(all_y, dim=0).cpu().numpy()

    smoothness_stats = {}

    for loss_name, (model, criterion) in models_dict.items():
        is_coral = 'coral' in loss_name.lower() or 'ordinal' in loss_name.lower()
        probs = get_prediction_probs(model, all_X, loss_name)

        # 获取预测 - 统一使用 criterion.predict() 与 evaluate.py 保持一致
        logits = model(all_X)
        if hasattr(criterion, 'predict'):
            preds = criterion.predict(logits).cpu().numpy()
        else:
            preds = logits.argmax(dim=1).cpu().numpy()

        total_violations = 0
        total_samples = len(probs)

        for i in range(total_samples):
            violations = check_probability_smoothness(probs[i], preds[i], is_coral=is_coral)
            total_violations += violations

        avg_violations = total_violations / total_samples

        # 额外统计：错误样本的平均级别差
        error_mask = preds != all_y
        if error_mask.sum() > 0:
            avg_error_diff = np.abs(preds[error_mask] - all_y[error_mask]).mean()
        else:
            avg_error_diff = 0.0

        smoothness_stats[loss_name] = {
            'avg_violations': avg_violations,
            'total_violations': total_violations,
            'total_samples': total_samples,
            'avg_error_diff': avg_error_diff
        }

    # 绘制统计图 - 两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = list(smoothness_stats.keys())
    violations = [smoothness_stats[n]['avg_violations'] for n in names]
    error_diffs = [smoothness_stats[n]['avg_error_diff'] for n in names]

    colors = ['#2ca02c' if 'coral' in n.lower() or 'ordinal' in n.lower() else '#d62728' for n in names]

    # 左图：违反指标
    bars1 = ax1.bar(names, violations, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Score (Lower is Better)', fontsize=11)
    ax1.set_xlabel('Loss Function', fontsize=11)
    ax1.set_title('Distribution Quality\n(CORAL: monotonicity | CE: cross-level tendency)',
                 fontsize=11, fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars1, violations):
        height = bar.get_height()
        if max(violations) > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(violations)*0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.tick_params(axis='x', rotation=15)

    # 右图：错误级别差
    bars2 = ax2.bar(names, error_diffs, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Average Grade Difference on Errors', fontsize=11)
    ax2.set_xlabel('Loss Function', fontsize=11)
    ax2.set_title('Ordinal Awareness\n(Lower = errors closer to truth)',
                 fontsize=11, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars2, error_diffs):
        height = bar.get_height()
        if max(error_diffs) > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(error_diffs)*0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax2.tick_params(axis='x', rotation=15)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='black', label='Ordinal Losses'),
        Patch(facecolor='#d62728', edgecolor='black', label='Standard Losses')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'smoothness_statistics.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[统计] 统计对比图已保存: {save_path}")

    # 打印统计结果
    print("\n模型对比统计:")
    print("-" * 65)
    print(f"{'Loss':<16} {'Dist Quality':>15} {'Error Grade Diff':>18}")
    print("-" * 65)
    for name, stats in smoothness_stats.items():
        print(f"{name:<16} {stats['avg_violations']:>15.4f} {stats['avg_error_diff']:>18.4f}")
    print("-" * 65)
