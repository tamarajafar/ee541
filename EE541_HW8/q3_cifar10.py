"""
EE 541 – HW 8  Problem 3: CIFAR-10 MLP Classification

Architecture:
  3072 → Linear(256) → ReLU → Dropout(0.3)
       → Linear(128) → ReLU → Dropout(0.3)
       → Linear(10)

L2 weight_decay = 1e-4, SGD with momentum, 50 epochs.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load CIFAR-10 ──────────────────────────────────────────────────────────────
# Download cifar-10-python.tar.gz from https://www.cs.toronto.edu/~kriz/cifar.html
# Extract so that ./data/cifar-10-batches-py/ exists.
CIFAR_ROOT = "./data/cifar-10-batches-py"

def unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")

def load_cifar10(root):
    Xs, ys = [], []
    for i in range(1, 6):
        d = unpickle(os.path.join(root, f"data_batch_{i}"))
        Xs.append(d[b"data"])
        ys.extend(d[b"labels"])
    Xtrain = np.concatenate(Xs, axis=0).astype(np.float32) / 255.0
    ytrain = np.array(ys, dtype=np.int64)

    d = unpickle(os.path.join(root, "test_batch"))
    Xtest = d[b"data"].astype(np.float32) / 255.0
    ytest = np.array(d[b"labels"], dtype=np.int64)
    return Xtrain, ytrain, Xtest, ytest

Xtrain, ytrain, Xtest, ytest = load_cifar10(CIFAR_ROOT)
print(f"Train: {Xtrain.shape}  Test: {Xtest.shape}")

# Per-channel normalisation
mean = Xtrain.reshape(-1, 3, 1024).mean(axis=(0, 2))  # (3,)
std  = Xtrain.reshape(-1, 3, 1024).std(axis=(0, 2))

def normalise(X):
    X = X.reshape(-1, 3, 1024)
    X = (X - mean[:, None]) / (std[:, None] + 1e-7)
    return X.reshape(-1, 3072)

Xtrain = normalise(Xtrain)
Xtest  = normalise(Xtest)

train_ds = TensorDataset(torch.tensor(Xtrain), torch.tensor(ytrain))
test_ds  = TensorDataset(torch.tensor(Xtest),  torch.tensor(ytest))
train_loader = DataLoader(train_ds, batch_size=100, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=100, shuffle=False, num_workers=0)

classes = ("plane", "car", "bird", "cat", "deer",
           "dog",   "frog", "horse", "ship", "truck")

# ── Model ──────────────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(3072, 256), nn.ReLU(), nn.Dropout(p=0.3),
    nn.Linear(256,  128), nn.ReLU(), nn.Dropout(p=0.3),
    nn.Linear(128,  10),
).to(device)

print(f"\nModel:\n{model}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)

# ── Eval helper ───────────────────────────────────────────────────────────────
def accuracy(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            correct += (model(X).argmax(1) == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total

# ── Training loop ─────────────────────────────────────────────────────────────
NUM_EPOCHS = 50
train_losses, test_losses = [], []
train_accs,   test_accs   = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

    tr_loss = running_loss / len(train_loader)
    tr_acc  = accuracy(train_loader)
    te_acc  = accuracy(test_loader)

    # compute test loss
    model.eval()
    te_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            te_loss += criterion(model(X), y).item()
    te_loss /= len(test_loader)

    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    train_accs.append(tr_acc)
    test_accs.append(te_acc)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
              f"Test  loss {te_loss:.4f} acc {te_acc:.2f}%")

print(f"\nFinal test accuracy: {test_accs[-1]:.2f}%")

# ── Learning curves ────────────────────────────────────────────────────────────
epochs = range(1, NUM_EPOCHS + 1)

fig1, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_losses, label="Train loss", linewidth=2)
ax.plot(epochs, test_losses,  label="Test loss",  linewidth=2, linestyle="--")
ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
ax.set_title("Problem 3 – CIFAR-10 Learning Curves", fontsize=13)
ax.legend(fontsize=12); ax.grid(True, alpha=0.4)
fig1.tight_layout()
fig1.savefig("p3_learning_curves.pdf", dpi=150)
plt.close(fig1)

fig2, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_accs, label="Train accuracy", linewidth=2)
ax.plot(epochs, test_accs,  label="Test accuracy",  linewidth=2, linestyle="--")
ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("Problem 3 – CIFAR-10 Accuracy vs Epoch", fontsize=13)
ax.legend(fontsize=12); ax.grid(True, alpha=0.4)
fig2.tight_layout()
fig2.savefig("p3_accuracy.pdf", dpi=150)
plt.close(fig2)

# ── Confusion matrix ───────────────────────────────────────────────────────────
model.eval()
all_p, all_y = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        all_p.extend(model(X).argmax(1).cpu().numpy())
        all_y.extend(y.numpy())

cm      = confusion_matrix(all_y, all_p)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig3, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=classes, yticklabels=classes,
            linewidths=0.4, linecolor="gray", ax=ax)
ax.set_xlabel("Predicted Label", fontsize=13)
ax.set_ylabel("True Label",      fontsize=13)
ax.set_title(f"Problem 3 – CIFAR-10 Confusion Matrix (row-normalised)\n"
             f"Test accuracy {test_accs[-1]:.2f}%", fontsize=13)
fig3.tight_layout()
fig3.savefig("p3_confusion_matrix.pdf", dpi=150)
plt.close(fig3)
print("Saved p3_confusion_matrix.pdf")

# ── Analysis: most confused class for each true class ─────────────────────────
print("\nMost confused class for each true class:")
for i, cls in enumerate(classes):
    row = cm_norm[i].copy()
    row[i] = 0  # exclude diagonal
    j = row.argmax()
    print(f"  {cls:8s}  →  most confused with  {classes[j]:8s}  "
          f"(confusion rate {cm_norm[i, j]:.3f})")

# Overall most confused pair
max_rate, pair = 0.0, (0, 0)
for i in range(10):
    for j in range(10):
        if i != j and cm_norm[i, j] > max_rate:
            max_rate = cm_norm[i, j]
            pair = (i, j)

print(f"\nMost confused pair overall: "
      f"true={classes[pair[0]]}, predicted={classes[pair[1]]} "
      f"(rate={max_rate:.3f})")
print("\nProblem 3 complete.")
