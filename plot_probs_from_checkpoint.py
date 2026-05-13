"""
从已保存的模型加载并生成预测概率折线图
无需重新训练，直接加载保存的模型权重
"""
import os
import torch
import numpy as np
from config import Config
from dataset import load_data
from model import MLPClassifier, CORALNet
from loss import get_loss
from plot_prediction_probs import plot_all_losses_probs


def load_model_from_checkpoint(model_path, model_type, input_dim, num_classes, device):
    """
    从检查点加载模型

    Args:
        model_path: 模型权重文件路径
        model_type: 'mlp' 或 'coral'
        input_dim: 输入维度
        num_classes: 类别数
        device: 设备
    """
    if model_type.lower() == 'coral':
        model = CORALNet(input_dim, Config.HIDDEN_DIMS, num_classes, Config.DROPOUT).to(device)
    else:
        model = MLPClassifier(input_dim, Config.HIDDEN_DIMS, num_classes, Config.DROPOUT).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def main():
    config = Config()

    # 加载数据
    print("加载数据...")
    train_loader, val_loader, test_loader, input_dim, num_classes, class_weights, le = load_data(config)
    print(f"Input dim: {input_dim}, Classes: {num_classes}")

    # 定义要加载的模型
    # 假设模型保存在 version_dir 下，名称为 best_{loss_name}.pt
    # 请根据实际情况修改路径
    version_dir = input("请输入版本目录路径 (例如: /home/ubuntu/lq/MLP_results/20250113_120000): ").strip()

    if not os.path.exists(version_dir):
        print(f"错误: 目录不存在: {version_dir}")
        return

    # 检测可用的模型
    available_losses = []
    for loss_name in ['ce', 'cdw_ce', 'cdw_ce_margin', 'mse', 'coral']:
        model_path = os.path.join(version_dir, f"best_{loss_name}.pt")
        if os.path.exists(model_path):
            available_losses.append(loss_name)

    if not available_losses:
        print("错误: 未找到任何已保存的模型!")
        return

    print(f"\n找到以下模型: {', '.join(available_losses)}")

    # 加载模型和对应的 loss
    models_dict = {}
    for loss_name in available_losses:
        model_path = os.path.join(version_dir, f"best_{loss_name}.pt")
        model_type = 'coral' if loss_name == 'coral' else 'mlp'
        model = load_model_from_checkpoint(model_path, model_type, input_dim, num_classes, config.DEVICE)
        criterion = get_loss(loss_name, num_classes=num_classes, device=config.DEVICE)
        models_dict[loss_name.upper()] = (model, criterion)
        print(f"已加载: {loss_name}")

    # 生成概率折线图
    label_names = [le.classes_[i] for i in range(len(le.classes_))]
    save_dir = os.path.join(version_dir, "probability_plots")

    print(f"\n生成预测概率折线图...")
    plot_all_losses_probs(
        models_dict=models_dict,
        test_loader=test_loader,
        device=config.DEVICE,
        label_names=label_names,
        save_dir=save_dir,
        num_samples=9
    )

    print(f"\n完成! 图像已保存至: {save_dir}")


if __name__ == "__main__":
    main()
