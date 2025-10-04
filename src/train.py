# train.py
"""
Train a vehicle classification model (PyTorch).

Usage example:
  python train.py --data_dir vehicle_dataset --epochs 20 --batch_size 32 --model efficientnet_b3

Outputs:
 - best_model.pth (saved in output_dir)
 - training_log.csv
 - confusion_matrix.png
"""

import os
import argparse
import numpy as np
from pathlib import Path
import copy
import time
import random
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="path to dataset folder (train/val)")
    p.add_argument("--output_dir", type=str, default="outputs", help="where to save models/logs")
    p.add_argument("--model", type=str, default="resnet50",
                   choices=["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Stronger augmentations for better generalization
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.2)], p=0.7),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # handle class imbalance
    targets = [s[1] for s in train_dataset.samples]
    class_sample_count = np.array([len(np.where(np.array(targets) == t)[0]) for t in np.unique(targets)])
    class_weights = 1.0 / (class_sample_count + 1e-6)
    samples_weight = np.array([class_weights[t] for t in targets])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_dataset.classes, class_weights


def build_model(model_name, num_classes, pretrained=True):
    if model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights="IMAGENET1K_V1" if pretrained else None)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError("Unknown model")
    return model


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, class_names, class_weights = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    num_classes = len(class_names)
    print("Found classes:", class_names)

    model = build_model(args.model, num_classes).to(device)

    cw = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    patience, patience_counter = 5, 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * imgs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += imgs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total

        # Validation
        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * imgs.size(0)
                val_corrects += torch.sum(preds == labels).item()
                val_total += imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_corrects / val_total
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={epoch_loss:.4f} acc={epoch_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        history.append({"epoch": epoch+1, "train_loss": epoch_loss, "train_acc": epoch_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "args": vars(args)
            }, os.path.join(args.output_dir, "best_model.pth"))
            print("✅ Saved best model (val_acc improved).")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹ Early stopping triggered!")
                break

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Classification report:\n", classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))

    # Save training history
    with open(os.path.join(args.output_dir, "training_log.csv"), "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    print("📊 Saved training log.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
