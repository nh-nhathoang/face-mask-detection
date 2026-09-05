"""
Fine-tune a ResNet18 classifier on the cropped face dataset produced by processing.py.

Handles the with_mask / without_mask / mask_weared_incorrect class imbalance
by weighting the loss inversely to class frequency.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

CROPS_DIR = "../data/crops"
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
SEED = 42
CHECKPOINT_PATH = "best_model.pth"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(crops_dir, batch_size, val_split, seed):
    """Split the ImageFolder dataset into stratified train/val loaders."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast = 0.2, saturation = 0.2),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        normalize,
    ])

    # Load twice with different transforms, since ImageFolder applies one
    # transform to the whole dataset and we only want augmentation on train.
    train_dataset = datasets.ImageFolder(crops_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(crops_dir, transform=val_transform)

    targets = train_dataset.targets  # same ordering in both, same underlying files
    indices = np.arange(len(targets))

    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=targets, random_state=seed
    )

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_targets = [targets[i] for i in train_idx]

    class_counts = np.bincount(train_targets)                                            
    sample_weights = [1.0 / class_counts[label] for label in train_targets]               
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)  

    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, train_dataset.classes, train_targets


def compute_class_weights(targets, num_classes):
    """Inverse-frequency weights so rare classes count more in the loss."""
    counts = np.bincount(targets, minlength=num_classes)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
