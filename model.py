import torch
import torch.nn as nn

# 标准的多层感知机，包含全连接层、批归一化和 ReLU 激活函数
class MLPClassifier(nn.Module):
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
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 它与 MLP 共享特征提取主干，但在输出端使用单个共享的线性层（输出 1 维）以及 K-1 个可学习的偏置参数，专门用于捕捉纤维化等级的有序递进关系。
class CORALNet(nn.Module):
    """
    CORALNet ordinal-regression head.
    Source: Saito et al., 2021.

    Shared feature backbone + single shared linear (output 1-dim)
    plus K-1 learnable biases. Output shape is [B, K-1] for K classes.
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
        self.feature = nn.Sequential(*layers)
        self.num_classes = num_classes
        self.shared = nn.Linear(prev_dim, 1)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        h = self.feature(x)
        z = self.shared(h)                     # [B, 1]
        logits = z + self.biases               # broadcast -> [B, K-1]
        return logits
