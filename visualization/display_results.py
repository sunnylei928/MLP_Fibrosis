"""
结果展示脚本
加载已保存的训练结果并展示图表和统计数据
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'liberation sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def display_results(version_dir):
    """
    展示指定版本的结果

    Args:
        version_dir: 版本目录路径，例如 /home/ubuntu/lq/MLP_results/20260513_035723
    """
    print(f"\n{'='*60}")
    print(f"结果展示: {version_dir}")
    print(f"{'='*60}")

    # 检查目录是否存在
    if not os.path.exists(version_dir):
        print(f"错误: 目录不存在: {version_dir}")
        return

    # 1. 显示日志文件内容
    log_file = os.path.join(version_dir, "training.log")
    if os.path.exists(log_file):
        print(f"\n{'='*60}")
        print("训练日志:")
        print(f"{'='*60}")
        with open(log_file, 'r', encoding='utf-8') as f:
            print(f.read())

    # 2. 显示结果统计
    results_file = os.path.join(version_dir, "results.json")
    if os.path.exists(results_file):
        print(f"\n{'='*60}")
        print("测试集结果:")
        print(f"{'='*60}")
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        print(f"{'Loss':<16} {'Acc':>8} {'AdjAcc':>8} {'MacroF1':>8} {'QWK':>8} {'MAE':>8}")
        print("-" * 65)
        for name, metrics in results.items():
            print(f"{name:<16} {metrics['accuracy']:>8.4f} {metrics['adjacent_accuracy']:>8.4f} "
                  f"{metrics['macro_f1']:>8.4f} {metrics['qwk']:>8.4f} {metrics['mae']:>8.4f}")

    # 3. 显示完整配置
    config_file = os.path.join(version_dir, "full_config.json")
    if os.path.exists(config_file):
        print(f"\n{'='*60}")
        print("完整配置:")
        print(f"{'='*60}")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        for key, value in config.items():
            if key != 'loss_configs':
                print(f"  {key}: {value}")

        if 'loss_configs' in config:
            print(f"\n  Loss 配置:")
            for loss_name, loss_info in config['loss_configs'].items():
                print(f"    {loss_name}: type={loss_info[0]}, weights={loss_info[1] is not None}")

    # 4. 列出生成的图表
    print(f"\n{'='*60}")
    print("生成的图表:")
    print(f"{'='*60}")

    image_files = []
    for file in os.listdir(version_dir):
        if file.endswith('.png'):
            image_files.append(file)

    if image_files:
        for file in sorted(image_files):
            file_path = os.path.join(version_dir, file)
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {file} ({size:.1f} KB)")

    # 检查 probability_plots 目录
    probs_dir = os.path.join(version_dir, "probability_plots")
    if os.path.exists(probs_dir):
        print(f"\n  概率可视化目录 (probability_plots/):")

        for root, dirs, files in os.walk(probs_dir):
            level = root.replace(probs_dir, '').count(os.sep)
            indent = ' ' * 2 * (level + 1)
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 2)
            for file in sorted(files):
                if file.endswith('.png'):
                    print(f"{subindent}{file}")

    # 检查 weights 目录
    weights_dir = os.path.join(version_dir, "weights")
    if os.path.exists(weights_dir):
        print(f"\n  模型权重目录 (weights/):")
        for file in sorted(os.listdir(weights_dir)):
            if file.endswith('.pt'):
                file_path = os.path.join(weights_dir, file)
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"    - {file} ({size:.1f} KB)")

    print(f"\n{'='*60}")


def list_all_results(base_dir):
    """列出所有可用的结果版本"""
    print(f"\n{'='*60}")
    print(f"可用结果版本: {base_dir}")
    print(f"{'='*60}")

    if not os.path.exists(base_dir):
        print(f"错误: 目录不存在: {base_dir}")
        return []

    versions = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item not in ['latest_run.txt']:
            # 获取修改时间
            mtime = os.path.getmtime(item_path)
            import datetime
            mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            versions.append((item, mtime_str, item_path))

    # 按修改时间排序
    versions.sort(key=lambda x: x[1], reverse=True)

    if versions:
        print(f"{'序号':<6} {'版本':<20} {'修改时间':<20}")
        print("-" * 50)
        for i, (name, mtime, path) in enumerate(versions[:20], 1):  # 只显示最近20个
            print(f"{i:<6} {name:<20} {mtime:<20}")

        return versions
    else:
        print("没有找到任何结果版本")
        return []


def main():
    import argparse
    parser = argparse.ArgumentParser(description='展示训练结果')
    parser.add_argument('--dir', type=str, help='结果目录路径')
    parser.add_argument('--list', action='store_true', help='列出所有可用版本')
    parser.add_argument('--base', type=str, default='/home/ubuntu/lq/MLP_results',
                       help='结果基础目录')

    args = parser.parse_args()

    if args.list:
        versions = list_all_results(args.base)
        if versions:
            print(f"\n使用 --dir <版本目录> 或 --dir <序号> 来查看具体结果")
    elif args.dir:
        # 检查是序号还是目录
        if args.dir.isdigit():
            versions = list_all_results(args.base)
            idx = int(args.dir) - 1
            if 0 <= idx < len(versions):
                display_results(versions[idx][2])
            else:
                print(f"错误: 序号 {args.dir} 超出范围")
        else:
            display_results(args.dir)
    else:
        # 默认显示最新的版本
        latest_file = os.path.join(args.base, "latest_run.txt")
        if os.path.exists(latest_file):
            with open(latest_file, 'r') as f:
                latest_dir = f.read().strip()
            display_results(latest_dir)
        else:
            print("请使用 --list 查看可用版本，或使用 --dir 指定版本")


if __name__ == "__main__":
    main()
