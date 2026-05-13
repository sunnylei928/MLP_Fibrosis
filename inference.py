"""
模型推理脚本 - 加载已有权重进行预测
支持：单模型推理、批量推理、结果导出
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import Config
from dataset import FibrosisDataset
from model import MLPClassifier, CORALNet
from utils import create_versioned_dir


def load_model_for_inference(weight_path, config, loss_type='coral'):
    """
    加载模型权重

    Args:
        weight_path: 权重文件路径
        config: 配置对象
        loss_type: loss类型 ('coral' 或其他)

    Returns:
        model: 加载好权重的模型
    """
    # 确定输入维度
    input_dim = len(config.NUMERIC_COLS) + len(config.CATEGORICAL_COLS) * 2  # 独热编码后
    num_classes = 5  # F0-F4

    # 创建模型
    if loss_type == 'coral':
        model = CORALNet(input_dim, config.HIDDEN_DIMS, num_classes, config.DROPOUT)
    else:
        model = MLPClassifier(input_dim, config.HIDDEN_DIMS, num_classes, config.DROPOUT)

    # 加载权重
    checkpoint = torch.load(weight_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()

    print(f"已加载权重: {weight_path}")
    print(f"  - Loss类型: {loss_type}")
    print(f"  - 最佳Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - 验证集准确率: {checkpoint.get('val_acc', 'N/A'):.4f}" if 'val_acc' in checkpoint else "")

    return model


def prepare_inference_data(excel_path, config):
    """
    准备推理数据

    Args:
        excel_path: 数据文件路径
        config: 配置对象

    Returns:
        X, y (如果有标签), le, scaler
    """
    # 读取数据
    df = pd.read_excel(excel_path)

    # 数据预处理（与训练时相同）
    df["HA"] = df["HA"].astype(str).str.replace(r"\.\.", ".", regex=True)
    df["HA"] = pd.to_numeric(df["HA"], errors="coerce")
    if df["HA"].isnull().sum() > 0:
        df["HA"] = df["HA"].fillna(df["HA"].median())

    # 独热编码
    df = pd.get_dummies(df, columns=config.CATEGORICAL_COLS, drop_first=False)

    # 如果有标签列，进行编码
    le = None
    if config.TARGET_COL in df.columns:
        le = LabelEncoder()
        df[config.TARGET_COL] = le.fit_transform(df[config.TARGET_COL])

    # 准备特征
    feature_cols = [c for c in df.columns
                    if c not in config.DROP_COLS + [config.TARGET_COL]]
    X = df[feature_cols].values.astype(np.float32)

    # 如果有标签
    y = None
    if config.TARGET_COL in df.columns:
        y = df[config.TARGET_COL].values.astype(np.int64)

    return X, y, le, df


def run_inference(model, X, config, batch_size=32):
    """
    运行推理

    Args:
        model: 模型
        X: 特征数据
        config: 配置
        batch_size: 批次大小

    Returns:
        preds: 预测标签
        probs: 概率 (coral时为累积概率)
    """
    dataset = FibrosisDataset(X, np.zeros(len(X)))  # 假标签
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(config.DEVICE)

            # 前向传播
            if isinstance(model, CORALNet):
                logits = model(batch_x)  # [B, K-1]
                probs = torch.sigmoid(logits)  # 累积概率

                # 转换为预测标签
                preds = []
                for prob in probs:
                    pred = torch.sum(prob > 0.5).item()
                    preds.append(pred)
                preds = torch.tensor(preds)
            else:
                logits = model(batch_x)  # [B, K]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    preds = torch.cat(all_preds).numpy()
    probs = torch.cat(all_probs).numpy()

    return preds, probs


def evaluate_predictions(preds, labels, le=None):
    """
    评估预测结果
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    acc = accuracy_score(labels, preds)
    adj_acc = np.mean(np.abs(labels - preds) <= 1)

    # F1分数
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')

    # QWK
    from sklearn.metrics import cohen_kappa_score
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')

    # MAE
    mae = np.mean(np.abs(labels - preds))

    results = {
        'accuracy': acc,
        'adjacent_accuracy': adj_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'qwk': qwk,
        'mae': mae,
    }

    print(f"\n预测结果评估:")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Adjacent Accuracy: {adj_acc:.4f}")
    print(f"  Macro F1:          {macro_f1:.4f}")
    print(f"  Weighted F1:       {weighted_f1:.4f}")
    print(f"  QWK:               {qwk:.4f}")
    print(f"  MAE:               {mae:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    print(f"\n混淆矩阵:")
    print(cm)

    return results


def main():
    parser = argparse.ArgumentParser(description='MLP Fibrosis Inference')
    parser.add_argument('--weight', type=str, required=True,
                        help='权重文件路径 (例如: weights/coral_best.pth)')
    parser.add_argument('--data', type=str, default=None,
                        help='推理数据路径 (默认使用config中的路径)')
    parser.add_argument('--loss_type', type=str, default='coral',
                        choices=['coral', 'ce', 'cdw_ce', 'mse'],
                        help='Loss类型 (决定使用哪个模型结构)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果保存目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')

    args = parser.parse_args()

    # 加载配置
    config = Config()

    # 设置数据路径
    data_path = args.data if args.data else config.DATA_PATH

    # 设置输出目录
    if args.output_dir:
        save_dir = args.output_dir
    else:
        save_dir = create_versioned_dir(config.OUTPUT_DIR + "_inference")

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MLP Fibrosis 推理")
    print(f"{'='*60}")
    print(f"权重文件: {args.weight}")
    print(f"数据文件: {data_path}")
    print(f"Loss类型: {args.loss_type}")
    print(f"保存目录: {save_dir}")

    # 准备数据
    print(f"\n准备数据...")
    X, y, le, df = prepare_inference_data(data_path, config)
    print(f"样本数: {len(X)}")
    print(f"特征数: {X.shape[1]}")

    # 标准化（需要使用训练时的scaler，这里简单处理用数据本身的均值方差）
    # 注意：实际应用中应该保存训练时的scaler参数
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 加载模型
    print(f"\n加载模型...")
    model = load_model_for_inference(args.weight, config, args.loss_type)

    # 运行推理
    print(f"\n运行推理...")
    preds, probs = run_inference(model, X, config, args.batch_size)

    # 保存预测结果
    results_df = df.copy()
    results_df['pred_label'] = preds

    # 保存标签名称（如果有label encoder）
    if le is not None:
        results_df['pred_class'] = le.inverse_transform(preds)
        if config.TARGET_COL in df.columns:
            results_df['true_class'] = le.inverse_transform(y)

    # 保存概率
    if isinstance(model, CORALNet):
        for i in range(probs.shape[1]):
            results_df[f'prob_P(y>{i})'] = probs[:, i]
    else:
        for i in range(probs.shape[1]):
            results_df[f'prob_class_{i}'] = probs[:, i]

    output_path = os.path.join(save_dir, 'predictions.xlsx')
    results_df.to_excel(output_path, index=False)
    print(f"\n预测结果已保存: {output_path}")

    # 如果有真实标签，计算评估指标
    if y is not None:
        metrics = evaluate_predictions(preds, y, le)

        # 保存评估结果
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型
            serializable = {k: float(v) for k, v in metrics.items()}
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  推理完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
