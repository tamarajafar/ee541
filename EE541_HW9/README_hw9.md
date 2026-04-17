# EE 541 Homework 9
Author: Tamara Jafar

Contents

q1/
├── q1_burning_liquid.py        # Main script: data prep, training, all plots
├── p1_learning_curves.pdf      # Log-loss curves with phase annotations
├── p1_accuracy_curves.pdf      # Accuracy curves with phase annotations
├── p1_confusion_matrix.pdf     # Row-normalised confusion matrix heatmap
├── p1_precision_recall.pdf     # Precision-recall curves (one-vs-rest, per class)
├── p1_baseline_vs_finetuned.pdf# Bar chart: baseline vs fine-tuned accuracy
├── p1_feature_maps_conv1.pdf   # Feature maps – first conv layer (64 filters)
├── p1_feature_maps_mid.pdf     # Feature maps – layer2[0].conv1 (64 of 128)
└── p1_feature_maps_deep.pdf    # Feature maps – layer4[0].conv1 (64 of 512)

## Dependencies
- Python 3.10+
- torch >= 2.0
- torchvision >= 0.15
- matplotlib
- seaborn
- scikit-learn
- numpy
- Pillow
