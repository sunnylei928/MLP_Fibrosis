"""
方案 1：修改 MLP 输出层以使用 CORAL loss
可行，但不是标准的 CORAL 方法
"""
import torch
import torch.nn as nn

class MLPForCORAL(nn.Module):
    """
    普通 MLP，但输出层改为 K-1 维以匹配 CORAL loss
    """
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        # 输出 K-1 个值，用于 CORAL loss
        layers.append(nn.Linear(prev_dim, num_classes - 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # [B, K-1]


# 使用方式
model = MLPForCORAL(input_dim=24, hidden_dims=[64, 32, 16], num_classes=5, dropout=0.3)
criterion = CORALNetLoss(num_classes=5)

# 训练时
logits = model(X)  # [B, 4]
loss = criterion(logits, y)
