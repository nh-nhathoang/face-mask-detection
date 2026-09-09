"""
Evaluate the trained mask classifier: confusion matrix + per-class precision/recall.

Reuses the same val split as training.py (same seed and val_split), so this
evaluates on data the model never trained on.
"""

import torch
from sklearn.metrics import classification_report, confusion_matrix

from training import (
    CROPS_DIR,
    VAL_SPLIT,
    SEED,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    build_dataloaders,
    build_model,
    get_device,
)


def load_trained_model(checkpoint_path, num_classes, device):
    model = build_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_predictions(model, loader, device):
    """Run inference over a loader and return true/predicted label lists."""
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu()
            preds = outputs.argmax(dim=1).cpu()

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            y_score.extend(probs.tolist())

    return y_true, y_pred, y_score


def evaluate_model(checkpoint_path=CHECKPOINT_PATH):
    """Convenience function for notebook use: returns everything needed to inspect results."""
    device = get_device()

    _, val_loader, classes, _, val_paths = build_dataloaders(CROPS_DIR, BATCH_SIZE, VAL_SPLIT, SEED)
    model = load_trained_model(checkpoint_path, len(classes), device)

    y_true, y_pred, y_score = get_predictions(model, val_loader, device)

    report = classification_report(y_true, y_pred, target_names=classes)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "classes": classes,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "val_paths": val_paths,
        "report": report,
        "confusion_matrix": cm,
    }
