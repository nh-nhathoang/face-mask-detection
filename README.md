 # Face Mask Detection

A computer vision project that detects whether people are wearing face masks correctly using deep learning for object detection.

## Project Overview

During the COVID-19 pandemic, masks became a fundamental thing for everyone. Monitoring mask usage in public places can help support compliance with safety guidelines.

This project uses the Face Mask Detection dataset from Kaggle to train an object detection model that identifies faces and classifies them into one of three categories:

- With Mask
- With Mask but worn incorrectly
- Without Mask

## Dataset

- **Source:** https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- **Annotation Format:** PASCAL VOC (XML)
- **Classes:**
  - with_mask
  - without_mask
  - mask_weared_incorrect

## Project Structure

```
Face-Mask-Detection/
│
├── data/
│   ├── images/
│   └── annotations/
├── notebooks/
├── src/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore
```
