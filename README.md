## Quick start

The .ipynb file is an independent standalone notebook. However more accuracy is obtained from the .py files compiled.

Run the whole pipeline (same order as the notebook) with:

```bash
python main.py
```

This runs: **train → eval → report → svm → pca → augment → predict**.

You can also run it explicitly:

```bash
python main.py all
```


# CIFAR-10 ResNet Small (Project)

This project is a cleaned-up **Python project version** of your notebook:
- Train a small ResNet-style CNN on **CIFAR-10**
- Evaluate on the official test set
- Print classification report + confusion matrix
- Train an **SVM baseline on CNN features**
- 3D PCA visualization of CNN features
- Compare original vs augmented image
- Predict a label for an external image URL

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

# Optional (interactive 3D PCA):
# pip install plotly
```

## Commands

Train (saves best model to `outputs/models/best_cifar10_model.keras`):
```bash
python main.py train --epochs 10 --batch-size 128
```

Evaluate the saved model:
```bash
python main.py eval
```

Classification report + confusion matrix:
```bash
python main.py report
```

SVM on CNN features:
```bash
python main.py svm
```

3D PCA visualization:
```bash
python main.py pca --n 6000
```

Compare original vs augmented image:
```bash
python main.py augment --index 56
```

Predict on an external image URL:
```bash
python main.py predict --url "https://www.lamborghini.com/sites/it-en/files/DAM/lamborghini/facelift_2019/homepage/families-gallery/2023/revuelto/revuelto_m.png"
```

## Outputs

- Models: `outputs/models/`
- Figures: `outputs/figures/`
  - `training_curves.png`
  - `confusion_matrix.png`
  - `augmentation_compare.png`
  - `pca_3d.html` (if Plotly works) or an image (matplotlib fallback)
