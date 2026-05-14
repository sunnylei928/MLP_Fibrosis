# MLP Fibrosis Classification

基于多层感知机的肝纤维化等级分类项目，支持多种损失函数和交叉验证评估。

## 项目结构

```
mlp_fibrosis/
├── config/              # 配置模块
│   └── __init__.py     # 全局配置
├── core/               # 核心模块
│   ├── dataset.py      # 数据集类
│   ├── model.py        # 模型定义 (MLP, CORAL)
│   ├── loss.py         # 损失函数
│   └── evaluate.py     # 评估指标
├── training/           # 训练模块
│   ├── train.py        # 训练循环
│   └── utils.py        # 训练工具
├── scripts/            # 执行脚本
│   ├── main.py                    # 单次训练
│   ├── main_kfold.py              # 五折交叉验证
│   ├── main_repeated_kfold.py    # 多次五折交叉验证 ★
│   ├── inference.py               # 推理脚本
│   └── plot_from_weights.py       # 权重评估绘图 ★
├── visualization/      # 可视化
│   ├── paper_figures.py          # 论文级图表 ★
│   ├── plot_prediction_probs.py  # 概率分布图
│   ├── display_results.py        # 结果展示
│   └── plot_probs_from_checkpoint.py
├── experiments/        # 实验代码
│   ├── kfold_cv.py
│   ├── repeated_kfold.py
│   └── test_mlp_coral.py
├── data/               # 数据目录
│   └── 深三数据useful_填充.xlsx
├── outputs/            # 输出目录
│   ├── weights/        # 模型权重
│   ├── logs/          # 训练日志
│   └── figures/       # 结果图表
└── requirements.txt    # 依赖列表
```

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行训练

**单次训练:**
```bash
python scripts/main.py
```

**五折交叉验证:**
```bash
python scripts/main_kfold.py
```

**多次五折交叉验证 (推荐):**
```bash
python scripts/main_repeated_kfold.py
```

### 从权重生成图表
```bash
python scripts/plot_from_weights.py
```

### 推理
```bash
python scripts/inference.py --weight outputs/weights/best_coral_r0_f0.pt --data data/深三数据useful_填充.xlsx
```

## 支持的损失函数

- **CE**: 标准交叉熵
- **CDW_CE**: 类别加权交叉熵
- **CDW_CE_MARGIN**: 带边际的类别加权交叉熵
- **MSE**: 均方误差
- **MLP_CORAL**: MLP + CORAL 损失 (简化有序回归)
- **CORAL**: CORALNet + CORAL 损失 (原始有序回归，共享权重)

## 模型架构

- **MLPClassifier**: 统一的多层感知机架构
  - 标准模式 (`ordinal_head=False`): 输出 5 类 logits
  - 有序模式 (`ordinal_head=True`): 输出 4 维 logits (K-1)，用于 CORAL 等 ordinal 损失
- **CORALNet**: 原始有序回归网络 (保留作为参考，实际训练使用 MLP)

## 评估指标

- Accuracy (准确率)
- Adjacent Accuracy (相邻准确率)
- Macro F1 / Weighted F1
- QWK (Quadratic Weighted Kappa)
- MAE (Mean Absolute Error)
