import torch

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
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    BATCH_SIZE = 32
    HIDDEN_DIMS = [256, 128, 64]
    # 设置小一点
    HIDDEN_DIMS = [256, 128, 64]
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 200
    PATIENCE = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = r"/home/ubuntu/lq/mlp_fibrosis/MLP_results"
