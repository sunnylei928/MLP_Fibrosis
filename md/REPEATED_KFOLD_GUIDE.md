# 多次五折交叉验证 - 完整指南

## 文件结构

```
mlp_fibrosis/
├── main.py                      # 单次训练 (train/val/test 一次性划分)
├── main_kfold.py                # 单次五折交叉验证
└── main_repeated_kfold.py       # 多次五折交叉验证 ⭐ 推荐
```

---

## 实验流程

### 单次训练 (main.py)
```
数据 → 一次划分 → 训练 → 测试 → 一个结果
```

### 五折交叉验证 (main_kfold.py)
```
数据 → 5次划分 → 5次训练 → 汇总(均值±标准差)
```

### 多次五折交叉验证 (main_repeated_kfold.py) ⭐
```
┌─────────────────────────────────────────────────┐
│ Run 1 (seed=42)                                   │
│   Fold1 → Fold2 → Fold3 → Fold4 → Fold5         │
│   → 汇总1 (5折的均值)                            │
├─────────────────────────────────────────────────┤
│ Run 2 (seed=123)                                  │
│   Fold1 → Fold2 → Fold3 → Fold4 → Fold5         │
│   → 汇总2 (5折的均值)                            │
├─────────────────────────────────────────────────┤
│ Run 3 (seed=456)                                  │
│   Fold1 → Fold2 → Fold3 → Fold4 → Fold5         │
│   → 汇总3 (5折的均值)                            │
└─────────────────────────────────────────────────┘
         ↓
总体汇总 (均值 ± 标准差 ± 95%置信区间)
```

---

## 使用方法

### 快速测试 (减少计算量)
```python
# 修改 main_repeated_kfold.py 中的参数
n_repeats = 2      # 只运行2次
seeds = [42, 123]    # 只用2个种子
```

### 标准实验 (推荐)
```python
n_repeats = 3
seeds = [42, 123, 456]
```

### 完整实验 (论文发表)
```python
n_repeats = 5
seeds = [42, 123, 456, 789, 1024]
```

### 运行
```bash
python main_repeated_kfold.py
```

---

## 输出文件

```
MLP_results/timestamp/
├── repeated_kfold.log              # 完整训练日志
├── repeated_kfold_summary.json     # 汇总统计
├── repeated_kfold_details.json     # 详细结果
├── repeated_kfold_comparison.png   # 对比图
├── experiment_report.md            # 文本报告 ⭐
└── weights/                        # 模型权重
    ├── ce_r0_f0.pt, ...
    └── coral_r2_f3.pt, ...
```

---

## 结果解读

### 1. 汇总统计格式
```
Loss      Accuracy              QWK                  MAE
──────────────────────────────────────────────────
CORAL     0.5234±0.0212 [0.4822,0.5646]  0.78±0.03 [0.72,0.84]  0.56±0.05
CE        0.4612±0.0345 [0.3945,0.5279]  0.71±0.05 [0.61,0.81]  0.68±0.07

均值±标准差 [95%置信区间]
```

### 2. 稳定性分析
```
Loss      五折变异    划分敏感度   评级
──────────────────────────────────────
CORAL     0.021      0.018       ✅ 优秀
CE        0.035      0.042       ⚠️ 一般
MSE       0.038      0.051       ❌ 较差

五折变异小 = 模型训练稳定
划分敏感度低 = 泛化能力强
```

### 3. 实验报告 (experiment_report.md)

自动生成 Markdown 格式报告，包含：
- 实验配置
- 结果汇总表格
- 稳定性分析
- 适合直接复制到论文

---

## 论文写作建议

### Results 部分写法

```
We performed 3-repeat 5-fold cross-validation to evaluate
model performance. The CORAL model achieved the best performance
with an accuracy of 52.34% (95% CI: [48.22%, 56.46%]), QWK of 0.78
(95% CI: [0.72, 0.84]), and MAE of 0.56 (95% CI: [0.51, 0.61]). See
Table X for detailed results.

Table X. Performance comparison under repeated k-fold cross-validation.
Values are mean ± std with 95% confidence intervals.

Method    Accuracy (%)        QWK                MAE
────────────────────────────────────────────────────
CORAL     52.34 ± 2.12        0.78 ± 0.03        0.56 ± 0.05
CE        46.12 ± 3.45        0.71 ± 0.05        0.68 ± 0.07
```

---

## 关键优势

| 特性 | 单次训练 | 五折CV | 多次五折CV |
|------|---------|-------|-----------|
| 评估可靠性 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 稳定性检验 | ❌ | ⚠️ | ✅ |
| 置信区间 | ❌ | ⚠️ | ✅ |
| 计算成本 | 低 | 中 | 高 |
| 论文认可度 | 低 | 高 | 最高 |

---

## 常见问题

**Q: 需要运行多少次？**
A: 至少 3次×5折 = 15次训练。建议使用标准配置 (3×5)。

**Q: 运行时间太长怎么办？**
A: 
1. 减少对比的 loss 数量（只保留 CE、CDW_CE、CORAL）
2. 减少 epoch 数（降低 PATIENCE）
3. 使用更少的隐藏层

**Q: 如何选择随机种子？**
A: 建议使用 [42, 123, 456] 或 [42, 123, 456, 789, 1024]。关键是种子之间要不同。

**Q: 结果变异很大怎么办？**
A: 说明模型对数据划分敏感，需要：
1. 增加正则化
2. 收集更多数据
3. 调整模型架构
