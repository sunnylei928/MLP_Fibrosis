import torch
import torch.nn as nn
import torch.nn.functional as F


class CDWCELoss(nn.Module):
    """
    Class Distance Weighted Cross-Entropy Loss.
    Source: Polat et al., *Class Distance Weighted Cross Entropy Loss
    for Classification of Disease Severity*, 2025.

    Core idea: the farther the predicted class is from the true class,
    the heavier the penalty.

    Basic formula:
        CDW-CE = -sum_{i=0}^{C-1} log(1 - p_i) * |i - c|^alpha

    where p_i = softmax(logits)_i, c is the ground-truth class index.
    When i = c the weight becomes 0, so correct predictions are not penalised.

    Margin variant (document formula has numerical issues; we clamp
    probabilities to [0, 1-eps] for stability):
        probs' = clamp(probs + margin, max=1-eps)
    """
    def __init__(self, alpha=1.0, margin=0.0, reduction='mean', eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)           # [B, C]
        num_classes = logits.size(-1)
        device = logits.device

        # Distance weights: |i - c|^alpha  [B, C]
        indices = torch.arange(num_classes, device=device).unsqueeze(0)
        c = targets.unsqueeze(1)
        weights = torch.abs(indices - c).float() ** self.alpha

        if self.margin > 0:
            probs = torch.clamp(probs + self.margin, max=1.0 - self.eps)

        loss_matrix = -torch.log(1.0 - probs + self.eps) * weights
        loss_per_sample = loss_matrix.sum(dim=-1)   # [B]

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        return loss_per_sample


class CORALNetLoss(nn.Module):
    """
    CORALNet ordinal-regression loss.
    Source: Saito et al., *Evaluation of ultrasonic fibrosis diagnostic
    system using convolutional network for ordinal regression*, 2021.

    Transforms a K-class ordinal problem into K-1 binary sub-problems.
    Input logits must have shape [B, K-1], produced by a network whose
    last layer uses **shared weights + K-1 independent biases**.

    At the loss level this is mathematically equivalent to CORN-style
    BCEWithLogitsLoss; the critical difference lies in the network head
    (see model.py).
    """
    def __init__(self, num_classes, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets):
        # logits: [B, K-1], targets: [B] in {0, ..., K-1}
        batch_size = logits.size(0)
        binary_targets = torch.zeros_like(logits)
        for i in range(batch_size):
            binary_targets[i, :targets[i]] = 1.0
        return self.bce(logits, binary_targets)

    @torch.no_grad()
    def predict(self, logits):
        """
        CORALNet inference: accumulate K-1 binary probabilities.
        pred = sum( sigmoid(logits_k) > 0.5 )
        """
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1).long()


class MSEClassificationLoss(nn.Module):
    """
    MSE adapted for a multi-class classifier with C-dim logits output.
    Converts hard labels to one-hot and computes MSE against softmax probs.
    This allows using MSE without changing the model architecture.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        num_classes = logits.size(-1)
        one_hot = F.one_hot(targets, num_classes=num_classes).float()
        return self.mse(probs, one_hot)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)
        one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        smoothed = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        return (-smoothed * log_probs).sum(dim=-1).mean()

class OrdinalCrossEntropy(nn.Module):
    """
    CORN-style ordinal regression loss.
    Treats K-class ordinal problem as K-1 binary classification tasks.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # logits: [B, num_classes-1]
        # targets: [B] in {0, 1, ..., num_classes-1}
        batch_size = logits.size(0)
        # Create binary labels: for class k, first k tasks are 1, rest are 0
        binary_targets = torch.zeros_like(logits)
        for i in range(batch_size):
            binary_targets[i, :targets[i]] = 1.0
        return self.bce(logits, binary_targets)


def get_loss(name, class_weights=None, num_classes=5, device='cpu',
             alpha=1.0, margin=0.05):
    """
    Factory for loss functions.

    New losses:
        'cdw_ce'        – CDW-CE (alpha controls distance-penalty strength)
        'cdw_ce_margin' – CDW-CE with additive margin
        'coral'         – CORALNet ordinal loss (expects logits [B, K-1])
        'mse'           – MSE on softmax probabilities vs one-hot targets
    """
    if name == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif name == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif name == 'focal':
        return FocalLoss(alpha=class_weights, gamma=2.0)
    elif name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(smoothing=0.1)
    elif name == 'ordinal':
        return OrdinalCrossEntropy(num_classes=num_classes)
    elif name == 'cdw_ce':
        return CDWCELoss(alpha=alpha, margin=0.0)
    elif name == 'cdw_ce_margin':
        return CDWCELoss(alpha=alpha, margin=margin)
    elif name == 'coral':
        return CORALNetLoss(num_classes=num_classes)
    elif name == 'mse':
        return MSEClassificationLoss()
    else:
        raise ValueError(f"Unknown loss: {name}")
