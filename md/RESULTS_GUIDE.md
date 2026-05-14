# MLP 纤维化分级项目 - 结果展示指南

## 目录结构

训练完成后，结果会保存在 `{OUTPUT_DIR}/{timestamp}/` 目录下，例如：
```
/home/ubuntu/lq/MLP_results/20260513_035723/
├── training.log              # 完整训练日志
├── full_config.json          # 完整配置信息
├── results.json              # 测试集结果统计
├── confusion_matrices.png    # 混淆矩阵对比图
├── metrics_comparison.png    # 指标对比柱状图
├── training_curves.png       # 训练曲线图
├── model_arch.txt            # 模型架构
├── meta.json                 # 运行元信息
├── weights/                  # 模型权重目录
│   ├── best_ce.pt
│   ├── best_cdw_ce.pt
│   ├── best_cdw_ce_margin.pt
│   ├── best_mse.pt
│   └── best_coral.pt
└── probability_plots/        # 概率可视化目录
    ├── probs_CE.png                    # CE 概率折线图
    ├── probs_CDW_CE.png
    ├── probs_CDW_CE_MARGIN.png
    ├── probs_MSE.png
    ├── probs_CORAL.png                 # CORAL 累积概率图
    ├── smoothness_statistics.png       # 平滑性统计对比
    └── ordinal_vs_standard/            # CORAL vs CE 对比
        ├── ordinal_vs_standard_1.png
        ├── ordinal_vs_standard_2.png
        └── ...
```

## 使用方法

### 1. 运行训练

```bash
python main.py
```

训练过程中：
- 所有输出会同时显示在终端和保存到 `training.log`
- 模型权重保存到 `weights/` 目录
- 训练完成后自动生成所有可视化图表

### 2. 查看结果

#### 方法一：查看最新结果
```bash
python display_results.py
```

#### 方法二：列出所有版本
```bash
python display_results.py --list
```

输出示例：
```
可用结果版本: /home/ubuntu/lq/MLP_results
==================================================
序号   版本                 修改时间
--------------------------------------------------
1      20260513_035723     2024-05-13 03:57:23
2      20260512_120545     2024-05-12 12:05:45
3      20260511_183022     2024-05-11 18:30:22
```

#### 方法三：查看指定版本
```bash
# 使用序号
python display_results.py --dir 1

# 或使用完整路径
python display_results.py --dir /home/ubuntu/lq/MLP_results/20260513_035723
```

### 3. 从已保存模型生成可视化

如果只想从已有模型生成概率可视化（不重新训练）：

```bash
python plot_probs_from_checkpoint.py
```

按提示输入版本目录路径即可。

## 结果文件说明

### 训练日志 (training.log)
包含完整的训练过程输出：
- 数据集信息
- 每个 epoch 的训练/验证指标
- 测试集评估结果
- 混淆矩阵和分类报告

### 完整配置 (full_config.json)
保存所有训练配置：
- 模型超参数（隐藏层维度、dropout 等）
- 训练参数（学习率、batch size 等）
- Loss 函数配置

### 模型权重 (weights/)
每个 loss 函数的模型权重单独保存：
```
weights/
├── best_ce.pt              # 使用 CE 训练的最佳模型
├── best_coral.pt           # 使用 CORAL 训练的最佳模型
└── ...
```

### 概率可视化 (probability_plots/)

#### 各模型概率折线图 (probs_*.png)
- **CE/MSE/CDW_CE**: 显示 5 个类别的概率分布
- **CORAL**: 显示 4 个累积概率 P(y>k)，应该单调递降

#### 平滑性统计图 (smoothness_statistics.png)
对比各模型的：
- 分布质量分数
- 错误级别差

#### CORAL vs CE 对比图 (ordinal_vs_standard/)
并排对比 CORAL 和 CE 在相同样本上的预测

## 指标说明

| 指标 | 含义 | 越高/低越好 |
|------|------|------------|
| Accuracy | 准确率 | 高 |
| Adjacent Accuracy | 相邻准确率（允许差1级） | 高 |
| Macro F1 | 宏平均 F1 | 高 |
| Weighted F1 | 加权 F1 | 高 |
| QWK | Quadratic Weighted Kappa | 高 |
| MAE | Mean Absolute Error | 低 |

**注意**：对于有序回归任务，QWK 和 MAE 是更重要的指标，因为它们考虑了类别之间的序数关系。
