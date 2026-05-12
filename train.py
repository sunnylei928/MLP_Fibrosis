import os
import torch
import numpy as np
from tqdm import tqdm
from evaluate import evaluate_epoch

class EarlyStopping:
    def __init__(self, patience=20, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = model.state_dict()
            self.counter = 0


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def train_model(model, train_loader, val_loader, criterion, optimizer, config,
                loss_name="ce", save_dir=None):
    early_stop = EarlyStopping(patience=config.PATIENCE)
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_adj_acc": []}

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        val_metrics = evaluate_epoch(model, val_loader, criterion, config.DEVICE)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_adj_acc"].append(val_metrics["adjacent_accuracy"])

        early_stop(val_metrics["loss"], model)

        if epoch % 20 == 0 or early_stop.early_stop:
            print(f"[{loss_name}] Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val Adj Acc: {val_metrics['adjacent_accuracy']:.4f}")

        if early_stop.early_stop:
            print(f"[{loss_name}] Early stopping at epoch {epoch}")
            break

    model.load_state_dict(early_stop.best_state)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f"best_{loss_name}.pt"))

    return history
