"""
快速启动脚本
"""
import sys
import os

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("  MLP Fibrosis Classification")
    print("=" * 60)
    print()
    print("请选择运行模式:")
    print("  1. 单次训练")
    print("  2. 五折交叉验证")
    print("  3. 多次五折交叉验证 (推荐)")
    print("  4. 从权重生成图表")
    print("  5. 推理")
    print()

    choice = input("请输入选项 (1-5): ").strip()

    if choice == "1":
        from scripts import main
        main.main()
    elif choice == "2":
        from scripts import main_kfold
        main_kfold.main()
    elif choice == "3":
        from scripts import main_repeated_kfold
        main_repeated_kfold.main()
    elif choice == "4":
        from scripts import plot_from_weights
        plot_from_weights.main()
    elif choice == "5":
        weight_path = input("权重文件路径: ").strip()
        data_path = input("数据路径 (回车使用默认): ").strip()
        loss_type = input("Loss类型 (mlp_coral/coral/ce/cdw_ce/mse): ").strip() or "coral"

        from scripts import inference
        import argparse

        args = argparse.Namespace(
            weight=weight_path,
            data=data_path if data_path else None,
            loss_type=loss_type,
            output_dir=None,
            batch_size=32
        )

        # 调用inference主函数
        import torch
        from config import Config
        from core.dataset import FibrosisDataset
        from core.model import MLPClassifier, CORALNet
        from training.utils import create_versioned_dir
        import pandas as pd
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        config = Config()

        # 准备数据
        df = pd.read_excel(args.data if args.data else config.DATA_PATH)
        df["HA"] = df["HA"].astype(str).str.replace(r"\.\..", ".", regex=True)
        df["HA"] = pd.to_numeric(df["HA"], errors="coerce")
        if df["HA"].isnull().sum() > 0:
            df["HA"] = df["HA"].fillna(df["HA"].median())
        df = pd.get_dummies(df, columns=config.CATEGORICAL_COLS, drop_first=False)
        le = LabelEncoder()
        df[config.TARGET_COL] = le.fit_transform(df[config.TARGET_COL])

        feature_cols = [c for c in df.columns
                        if c not in config.DROP_COLS + [config.TARGET_COL]]
        X = df[feature_cols].values.astype(np.float32)
        y = df[config.TARGET_COL].values.astype(np.int64)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # 加载模型
        input_dim = X.shape[1]
        num_classes = 5

        # 根据损失类型选择架构
        if args.loss_type == 'coral':
            # CORALNet + CORAL loss (原始架构)
            model = CORALNet(input_dim, config.HIDDEN_DIMS, num_classes, config.DROPOUT)
        elif args.loss_type == 'mlp_coral':
            # MLP + CORAL loss (简化架构)
            model = MLPClassifier(input_dim, config.HIDDEN_DIMS, num_classes,
                                  config.DROPOUT, ordinal_head=True)
        else:
            # 标准分类损失
            model = MLPClassifier(input_dim, config.HIDDEN_DIMS, num_classes,
                                  config.DROPOUT, ordinal_head=False)

        checkpoint = torch.load(args.weight, map_location=config.DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(config.DEVICE)
        model.eval()

        # 推理
        from torch.utils.data import DataLoader
        test_dataset = FibrosisDataset(X, y)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(config.DEVICE)
                if args.loss_type == 'coral':
                    logits = model(batch_x)
                    probs = torch.sigmoid(logits)
                    preds = torch.sum(probs > 0.5, dim=1)
                else:
                    logits = model(batch_x)
                    preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())

        preds = torch.cat(all_preds).numpy()

        print(f"\n预测完成! 共 {len(preds)} 个样本")
        print(f"预测分布: {np.bincount(preds)}")

    else:
        print("无效选项!")


if __name__ == "__main__":
    main()
