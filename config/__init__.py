"""
配置模块
"""
import torch
import os

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """全局配置"""

    # 数据路径
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "深三数据useful_填充.xlsx")

    # 数据列配置
    TARGET_COL = "LABLE_F"
    DROP_COLS = ["编号", "姓名"]
    CATEGORICAL_COLS = ["性别", "Machine"]
    NUMERIC_COLS = [
        "年龄", "身高", "体重", "BMI", "PLT", "ALT", "AST", "GGT",
        "ALB", "ALP", "HA", "PIIIP", "CIV", "LN", "AST/PLT",
        "AST/ALT", "P-SWE", "2D-SWE", "ARPI（AST/40）", "FIB-4"
    ]

    # 训练参数
    RANDOM_SEED = 42
    TEST_SIZE = 0.3
    VAL_SIZE = 0.2
    BATCH_SIZE = 32
    HIDDEN_DIMS = [64, 32, 16]
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 200
    PATIENCE = 30

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出目录
    OUTPUT_DIR = r"/home/ubuntu/lq/MLP_results" 


# 导出
__all__ = ['Config']
