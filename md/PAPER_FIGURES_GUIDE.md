# 高级论文图表绘制指南

## 论文图表设计原则

### 1. 清晰可读
- 字体大小 ≥ 8pt (单栏图) 或 ≥ 6pt (双栏图)
- 线条粗细适中 (1.0-1.5)
- 避免杂乱元素

### 2. 色盲友好
- 使用色盲友好配色
- 不仅用颜色区分，加用形状/线型
- 高对比度

### 3. 统计严谨
- 包含误差条 (标准差)
- 标注置信区间
- 样本量标注

### 4. 符合期刊要求
- DPI ≥ 300
- 矢量图格式 (PDF/EPS)
- 单栏/双栏尺寸适配

---

## 高级论文常用图表类型

### 1. 分组柱状图 + 误差条
**用途**: 主要结果展示
**期刊示例**: Nature, JAMA, Lancet

```
关键要素:
✓ 分组对比多个指标
✓ 误差条表示标准差
✓ 垂线表示置信区间
✓ 数值标签
✓ 图例清晰
```

### 2. 箱线图 (Box Plot)
**用途**: 展示分布和异常值
**期刊示例**: Nature Methods, Bioinformatics

```
关键要素:
✓ 箱体: 四分位数
✓ 中线: 中位数
✓ 需须: 异常值
✅ 菱形: 均值
```

### 3. 森林图 (Forest Plot)
**用途**: Meta分析
**期刊示例**: NEJM, Lancet, BMJ

```
关键要素:
✓ 点估计: 均值
✓ 横线: 置信区间
✓ 参考线: 无效线
✓ 排序: 按效应大小
```

### 4. 雷达图 (Radar Chart)
**用途**: 多维度对比
**期刊示例**: IEEE Transactions

```
关键要素:
✅ 多维度展示
✅ 面积表示综合能力
✅ 适合对比3-5个模型
```

### 5. 热图 (Heatmap)
**用途: 相关性/性能矩阵
**期刊示例**: Nature Genetics

```
关键要素:
✓ 颜色深浅表示值大小
✓ 添加数值标签
✓ 色条说明
```

---

## Nature/Science 级图表技巧

### 技巧 1: 分面板布局
```
┌─────────────────────────────────┐
│ Panel A: 主要结果                 │
│ ┌─────┬─────┬─────┐             │
│ │  图  │  图  │  图  │             │
│ └─────┴─────┴─────┘             │
│ Panel B: 补充结果                 │
│ ┌───────────────────────────┐   │
│ │  图                      │   │
│ └───────────────────────────┘   │
└─────────────────────────────────┘
```

### 技巧 2: 一致的颜色方案
```
统一配色方案:
CORAL  - 绿绿色 (#009E73)
CE     - 橙色   (#D55E00)
MSE    - 蓝色   (#0072B2)

每个图使用相同颜色
```

### 技巧 3: 统计信息完整
```
每个图表包含:
- 均值 (Mean)
- 标准差 (SD)
- 置信区间 (95% CI)
- 样本量 (n=75)
```

### 技巧 4: 高 DPI 输出
```python
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

---

## 顶级期刊图表对比

### 期刊要求

| 期刊 | DPI | 字体 | 格式 | 推荐图表 |
|------|-----|------|------|----------|
| Nature | 300-600 | Arial/Helvetica | PDF/TIF | 分组柱状图、热图 |
| Science | 300 | Arial | PDF | 简洁图表、大字体 |
| Lancet | 300 | Times | PDF | 森林图、箱线图 |
| IEEE | 300 | Times New Roman | PDF/EPS | 工程图、雷达图 |

---

## 实用代码模板

### 模板 1: 分组柱状图 (最常用)
```python
def plot_grouped_bar(summary):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    loss_names = ['CE', 'CORAL', 'MSE']
    x_pos = np.arange(len(loss_names))
    width = 0.25
    
    # 绘制分组
    for i, metric in enumerate(['Accuracy', 'QWK', 'MAE']):
        means = [summary[ln][metric] for ln in loss_names]
        stds = [summary[ln][f'{metric}_std'] for ln in loss_names]
        
        offset = (i - 1) * width
        ax.bar(x_pos + offset, means, width, yerr=stds,
               label=metric, color=COLORS[i], alpha=0.8)
    
    ax.legend()
    ax.set_ylabel('Score')
    plt.tight_layout()
    plt.savefig('figure.png', dpi=300)
```

### 模板 2: 森林图 (Meta分析风格)
```python
def plot_forest(summary):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 数据准备
    models = ['CE', 'CORAL', 'MSE']
    means = [0.46, 0.52, 0.48]
    ci_low = [0.40, 0.48, 0.42]
    ci_high = [0.52, 0.56, 0.54]
    
    y_pos = np.arange(len(models))
    
    # 绘制
    ax.errorbar(means, y_pos, 
                xerr=[np.array(means)-np.array(ci_low),
                      np.array(ci_high)-np.array(means)],
                fmt='o', capsize=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Accuracy')
    plt.tight_layout()
```

---

## 你的项目对应的图表

### 必需图表 (论文 Figure 1)
1. **性能对比柱状图**
   - X轴: CE, CDW_CE, CORAL, MSE
   - Y轴分组: Accuracy, QWK, MAE
   - 误差条: 标准差
   - 置信区间: 垂线标注

### 补充图表
2. **稳定性分析图** (Figure 2)
   - 五折内变异 vs 运行间变异
   - 变异比指标

### 可选图表
3. **热图**: 整体性能矩阵
4. **箱线图**: 分布展示

---

## 快速生成

使用 `paper_figures.py` 中的 `PaperFigures` 类:

```python
from paper_figures import PaperFigures

# 运行多次五折交叉验证后
# summary = {...}  # 从 main_repeated_kfold.py 获取

figures = PaperFigures(save_dir)
figures.plot_all(summary)  # 生成所有图表
```

输出文件:
- figure1_performance_comparison.png
- figure7_stability.png
- figure_legend.md

---

## 常见错误避免

❌ **错误做法**:
- 不标误差条
- 不标注样本量
- 使用红绿色 (色盲无法区分)
- 分辨率太低 (< 150 DPI)
- 字体太小 (< 6pt)

✅ **正确做法**:
- 标注误差或置信区间
- 标注 n=75
- 使用色盲友好配色
- DPI ≥ 300
- 字体 ≥ 8pt
