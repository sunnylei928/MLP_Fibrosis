import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             cohen_kappa_score, mean_absolute_error) # 增加这两个

@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        if hasattr(criterion, 'predict'):
            preds = criterion.predict(logits)
        else:
            preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    # Adjacent accuracy: prediction within 1 grade of ground truth
    adj_acc = np.mean(np.abs(all_preds - all_labels) <= 1)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    # 新增计算 QWK 和 MAE
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    mae = mean_absolute_error(all_labels, all_preds)

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": acc,
        "adjacent_accuracy": adj_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "qwk": qwk,  # 新增返回
        "mae": mae,  # 新增返回
        "preds": all_preds,
        "labels": all_labels
    }


def print_report(metrics, label_encoder, title="Test"):
    print(f"\n{'='*50}")
    print(f"  {title} Results")
    print(f"{'='*50}")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  Adjacent Accuracy:   {metrics['adjacent_accuracy']:.4f}")
    print(f"  Macro F1:            {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:         {metrics['weighted_f1']:.4f}")
    print(f"  QWK (Kappa):         {metrics['qwk']:.4f}")
    print(f"  MAE:                 {metrics['mae']:.4f}")
    print(f"{'='*50}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(metrics["labels"], metrics["preds"]))

    print("\nClassification Report:")
    target_names = [label_encoder.classes_[i] for i in range(len(label_encoder.classes_))]
    print(classification_report(metrics["labels"], metrics["preds"],
                                target_names=target_names, digits=4))
