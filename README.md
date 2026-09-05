 # Face Mask Detection

A computer vision project that detects whether people are wearing face masks correctly using CNN for object detection.

## Project Overview

During the COVID-19 pandemic, masks became a fundamental thing for everyone. Monitoring mask usage in public places can help support compliance with safety guidelines.

This project uses the Face Mask Detection dataset from Kaggle to train an object detection model that identifies faces and classifies them into one of three categories:

- With Mask
- With Mask but worn incorrectly
- Without Mask

## Dataset

- **Source:** https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- **Annotation Format:** PASCAL VOC (XML)
- **Images:** 853 (with 4,072 labeled faces in total)
- **Classes distributions by faces (not images):**
  - with_mask: 3,232
  - without_mask: 717
  - mask_weared_incorrect: 123

## Project Structure

```
Face-Mask-Detection/
│
├── data/
│   ├── images/
│   ├── crops/
│   └── annotations/
├── src/
│   ├── utils.py
│   ├── processing.py
│   ├── training.py
│   ├── evaluation.py
│   └── main.ipynb
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore
```
The optimization are based on the severe imbalance in the minor class (about 26:1 against the largest class):

## Approach

1. **EDA** (`notebooks/EDA.ipynb`) — parses annotations, checks class distribution, face-size statistics, and faces-per-image, and visualizes sample bounding boxes.
2. **Face cropping** (`src/processing.py`) — crops each labeled face out of its source image using its bounding box, saving into class-named folders (`with_mask/`, `without_mask/`, `mask_weared_incorrect/`) for classification-style loading via `ImageFolder`. Boxes below a minimum size are skipped as degenerate.
3. **Model** (`src/training.py`) — ResNet18, pretrained on ImageNet, with the final layer replaced for 3-class output. Images are resized to 128x128 and normalized with ImageNet statistics. Train/val split is stratified (80/20) to preserve class ratios in both sets.
4. **Handling class imbalance** — two approaches were tried:
   - Inverse-frequency loss weighting (`compute_class_weights`), which worked but proved sensitive to overcorrection — an overly aggressive learning rate combined with it caused recall on the rare class to rise at a steep cost to precision.
   - A `WeightedRandomSampler` on the training loader, oversampling the rare class instead of reweighting its loss. This produced the best-balanced result (see below) without the loss-weighting approach's precision trade-off, and is the current default.
5. **Regularization** — `weight_decay=1e-4` on the Adam optimizer, plus checkpointing on the best validation loss (validation loss is computed with a separate, unweighted criterion — using the same weighted criterion for both training and validation distorted the validation signal, since the rare class's ~10x weight was inconsistently represented across small validation batches).
6. **Data augmentation** — horizontal flip on all training crops. Heavier, class-targeted augmentation (rotation, color jitter, applied only to the rare class) was also tested, but destabilized training and roughly halved recall on that class — this was reverted, and is noted below as a negative result rather than left undocumented.
7. **Evaluation** (`src/evaluation.py`) — classification report (precision/recall/F1 per class) and confusion matrix on the validation subset.
