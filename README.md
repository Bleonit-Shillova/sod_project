
# Salient Object Detection (SOD) Project

## Overview
This project implements a complete Salient Object Detection pipeline using UNet models from scratch.

## Dataset
- DUTS (TR + TE)
- Resized to 128×128
- Normalized 0–1
- 70/15/15 split
- Augmentations: flip, color jitter

## Models
- Simple UNet
- Dropout UNet
- Deep UNet

## Results
| Model | IoU | Precision | Recall | F1 | MAE |
|-------|------|-----------|---------|--------|-------|
| Simple UNet | 0.7617 | 0.8542 | 0.8691 | 0.8538 | 0.0777 |
| Dropout UNet | 0.7603 | 0.8612 | 0.8611 | 0.8531 | 0.0782 |
| Deep UNet | 0.7509 | 0.8632 | 0.8459 | 0.8464 | 0.0771 |

## Demo
Notebook supports:
- Image upload
- Inference
- Overlay heatmap
- Inference speed

## Files
- train.py
- evaluate.py
- data_loader.py
- sod_model.py
- sod_model_dropout.py
- sod_model_deep.py
- demo_notebook.ipynb

