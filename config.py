import torch


'''优化器：使用 AdamW

学习率 (Learning Rate)：1e-3

权重衰减 (Weight Decay)：1e-4

Batch Size：32

迭代次数 (Epochs)：最大训练 200 个 Epoch

早停机制 (Patience)：设置为 30，即验证集指标连续 30 个 Epoch 没有提升时将提前停止训练

数据集划分：测试集占 20% (0.2)，验证集占 10% (0.1)，且固定了随机种子为 42

隐藏层维度：统一设置为 [256, 128, 64]

正则化：每层都应用了 BatchNorm1d 和丢弃率为 0.3 的 Dropout'''
# 目标预测列：LABLE_F（即纤维化等级）。

# 分类特征：性别 (Gender) 和 Machine (超声仪器型号)。

# 数值特征：共 20 个，涵盖了基础体征（如年龄、BMI）、常规血液和肝功能指标（如 AST、ALT、PLT 等），以及肝脏硬度/超声指标（如 2D-SWE、P-SWE）和组合指数（FIB-4、ARPI）。

# HIDDEN_DIMS = [256, 128, 64]当前的模型包含 4层全连接层（即 3 个隐藏层 + 1 个输出层）。
 
# 通过循环为这 3 个维度分别创建一层网络，每一层都包含：

# 线性层 (Linear)：对应维度为 输入层->256, 256->128, 128->64

# 批归一化 (BatchNorm1d)

# 激活函数 (ReLU)

# Dropout (设为 0.3)

# 输出层（一层）：在基础分类器（MLPClassifier）中，最后一层是从 64 维映射到类别数（num_classes）的线性层。

# 在有序回归模型（CORALNet）中，最后一层是从 64 维映射到 1 维的共享线性层（并加上对应等级数量的偏置参数）。


class Config:
    DATA_PATH = r"/home/ubuntu/lq/mlp_fibrosis/深三数据useful_填充.xlsx"
    TARGET_COL = "LABLE_F"
    DROP_COLS = ["编号", "姓名"]
    CATEGORICAL_COLS = ["性别", "Machine"]
    NUMERIC_COLS = [
        "年龄", "身高", "体重", "BMI", "PLT", "ALT", "AST", "GGT",
        "ALB", "ALP", "HA", "PIIIP", "CIV", "LN", "AST/PLT",
        "AST/ALT", "P-SWE", "2D-SWE", "ARPI（AST/40）", "FIB-4"
    ]
    RANDOM_SEED = 42
    TEST_SIZE = 0.3
    VAL_SIZE = 0.2
    BATCH_SIZE = 32
    # HIDDEN_DIMS = [256, 128, 64]
    HIDDEN_DIMS = [64, 32, 16]
    # HIDDEN_DIMS = [32, 16, 8]
    # HIDDEN_DIMS = [64, 32, 16, 8]
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 200
    PATIENCE = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = r"/home/ubuntu/lq/MLP_results"
