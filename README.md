
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
| Simple UNet | 0.7478 | 0.87724 | 0.8283 | 0.8437 | 0.0777 |
| Dropout UNet | 0.7611 | 0.8616 | 0.8616 | 0.8536 | 0.0782 |
| Deep UNet | 0.7517 | 0.8637 | 0.8464 | 0.84704 | 0.0771 |

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


## 📥 Dataset Download (Official Links)

This project uses the **DUTS dataset** for Salient Object Detection.

You can download the original dataset here:

- **DUTS-TR (Training Set)**  
  https://saliencydetection.net/duts/download/DUTS-TR.zip

- **DUTS-TE (Testing Set)**  
  https://saliencydetection.net/duts/download/DUTS-TE.zip

After downloading, extract both zips inside a folder and run the dataset preparation 
script included in this project (Colab notebook automatically handles it).
