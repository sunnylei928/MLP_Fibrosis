import os
import torch
import numpy as np
from core.evaluate import evaluate_epoch

class EarlyStopping:
    def __init__(self, patience=20, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, monitor_value, model):
        # 监控指标（如 MAE）越小越好，故取负值转为 score
        score = -monitor_value
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        # 使用 .cpu().clone() 确保保存的是当前权重的副本
        self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}


def train_epoch(model, loader, optimizer, criterion, device):
    """保持独立，负责单轮训练的梯度更新"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for X, y in loader:
        # 跳过大小为1的batch（BatchNorm要求每个channel至少有2个值）
        if X.size(0) < 2:
            continue
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)
    return total_loss / total_samples if total_samples > 0 else 0.0


def train_model(model, train_loader, val_loader, criterion, optimizer, config,
                loss_name="ce", save_dir=None):
    
    # 1. 初始化早停和调度器，统一监控 MAE
    early_stop = EarlyStopping(patience=config.PATIENCE)
    
    # 监控验证集 MAE：5轮不降则 LR 减半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6
    )

    history = {
        "train_loss": [], "val_loss": [], "val_acc": [], "val_adj_acc": [],
        "val_mae": [], "val_qwk": []
    }


    for epoch in range(1, config.EPOCHS + 1):
        # --- 调用独立的 train_epoch ---
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        
        # --- 调用 evaluate_epoch 获取指标 ---
        val_metrics = evaluate_epoch(
            model, val_loader, criterion, config.DEVICE
        )

        # 记录数据
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_adj_acc"].append(val_metrics["adjacent_accuracy"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_qwk"].append(val_metrics["qwk"])

        # 2. 核心联动：根据当前 MAE 更新学习率和早停状态
        current_mae = val_metrics['mae']
        scheduler.step(current_mae)
        early_stop(current_mae, model)

        # 3. 打印进度（包含实时学习率 LR）
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 10 == 0 or early_stop.early_stop:
            print(f"[{loss_name}] Epoch {epoch:03d} | LR: {current_lr:.2e} | "
                  f"MAE: {current_mae:.4f} | QWK: {val_metrics['qwk']:.4f}")

        if early_stop.early_stop:
            print(f"[{loss_name}] 触发早停！最佳验证集 MAE: {-early_stop.best_score:.4f}")
            break

    # 4. 训练结束，恢复表现最好的模型参数
    model.load_state_dict(early_stop.best_state)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f"best_{loss_name}.pt"))

    return history