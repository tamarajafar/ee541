"""
EE 541 - Homework 8
Problem 1: PyTorch Logistic Classifier (MNIST)

Single-layer linear classifier trained with mini-batch SGD and L2 regularization.
Reads from HDF5 files mnist_traindata.hdf5 / mnist_testdata.hdf5.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Dataset ──────────────────────────────────────────────────────────────────
TRAIN_PATH = "/mnt/user-data/uploads/mnist_traindata.hdf5"
TEST_PATH  = "/mnt/user-data/uploads/mnist_testdata.hdf5"

class MNISTHDF5Dataset(Dataset):
    """
    Reads images (xdata) and one-hot labels (ydata) from the HDF5 files
    provided for EE 541.  Labels are stored as one-hot vectors so we
    convert them to integer class indices.
    """
    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
        # xdata: (N, 784) float32, already flattened
        self.images = self.file["xdata"][:]          # load fully into RAM
        # ydata: (N, 10) one-hot float64 → integer labels
        self.labels = np.argmax(self.file["ydata"][:], axis=1).astype(np.int64)
        self.file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = int(self.labels[idx])
        return image, label


# ── Hyper-parameters ─────────────────────────────────────────────────────────
BATCH_SIZE   = 100
NUM_EPOCHS   = 30
LR           = 0.1
WEIGHT_DECAY = 1e-4   # L2 regularisation coefficient

INPUT_SIZE   = 784    # 28 × 28
NUM_CLASSES  = 10

# ── Data loaders ─────────────────────────────────────────────────────────────
train_dataset = MNISTHDF5Dataset(TRAIN_PATH)
test_dataset  = MNISTHDF5Dataset(TEST_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset):,}  |  Test samples: {len(test_dataset):,}")
print(f"Batches per epoch: {len(train_loader)}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(INPUT_SIZE, NUM_CLASSES)   # single fully-connected layer (logistic)
).to(device)

print(f"\nModel:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ── Loss & optimiser ─────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
# SGD with L2 weight_decay for regularisation
optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ── Helper: evaluate one full pass ───────────────────────────────────────────
def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)
            total_loss += loss.item()
            preds  = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return total_loss / len(loader), 100.0 * correct / total

# ── Training loop ─────────────────────────────────────────────────────────────
train_losses, test_losses = [], []
train_accs,   test_accs   = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)

    tr_loss, tr_acc = evaluate(train_loader)
    te_loss, te_acc = evaluate(test_loader)

    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    train_accs.append(tr_acc)
    test_accs.append(te_acc)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train loss {tr_loss:.4f}  acc {tr_acc:.2f}% | "
              f"Test  loss {te_loss:.4f}  acc {te_acc:.2f}%")

print(f"\nFinal test accuracy: {test_accs[-1]:.2f}%")

# ── Figure 1: Learning curves (log-loss) ──────────────────────────────────────
epochs = range(1, NUM_EPOCHS + 1)
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(epochs, train_losses, label="Train loss", linewidth=2)
ax1.plot(epochs, test_losses,  label="Test loss",  linewidth=2, linestyle="--")
ax1.set_xlabel("Epoch", fontsize=13)
ax1.set_ylabel("Cross-Entropy Loss", fontsize=13)
ax1.set_title("Problem 1 – Learning Curves (Log-Loss)\n"
              f"SGD  lr={LR}  λ_L2={WEIGHT_DECAY}  batch={BATCH_SIZE}", fontsize=13)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.4)
fig1.tight_layout()
fig1.savefig("/home/claude/hw8/q1/p1_learning_curves.pdf", dpi=150)
plt.close(fig1)
print("Saved: p1_learning_curves.pdf")

# ── Figure 2: Accuracy curves ─────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(epochs, train_accs, label="Train accuracy", linewidth=2)
ax2.plot(epochs, test_accs,  label="Test accuracy",  linewidth=2, linestyle="--")
ax2.set_xlabel("Epoch", fontsize=13)
ax2.set_ylabel("Accuracy (%)", fontsize=13)
ax2.set_title("Problem 1 – Accuracy vs Epoch", fontsize=13)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.4)
fig2.tight_layout()
fig2.savefig("/home/claude/hw8/q1/p1_accuracy.pdf", dpi=150)
plt.close(fig2)
print("Saved: p1_accuracy.pdf")

# ── Confusion matrix ──────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        logits = model(X)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # row-normalise

fig3, ax3 = plt.subplots(figsize=(9, 7))
sns.heatmap(
    cm_norm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=range(10), yticklabels=range(10),
    linewidths=0.4, linecolor="gray", ax=ax3
)
ax3.set_xlabel("Predicted Label", fontsize=13)
ax3.set_ylabel("True Label",      fontsize=13)
ax3.set_title("Problem 1 – Confusion Matrix (row-normalised)\n"
              f"Test accuracy {test_accs[-1]:.2f}%", fontsize=13)
fig3.tight_layout()
fig3.savefig("/home/claude/hw8/q1/p1_confusion_matrix.pdf", dpi=150)
plt.close(fig3)
print("Saved: p1_confusion_matrix.pdf")
print("\nProblem 1 complete.")
