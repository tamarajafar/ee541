"""
EE 541 - Homework 8
Problem 2: Regularization and Dropout (Fashion MNIST)

Model A: 1 hidden layer, 128 nodes, ReLU. No regularisation, no dropout.
Model B: 1 hidden layer,  48 nodes, ReLU. L2 λ=1e-4, dropout p=0.2.

Trains both for 40 epochs, then produces weight histograms.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Fashion MNIST ─────────────────────────────────────────────────────────────
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.FashionMNIST(root="./data", train=True,  download=False, transform=transform)
test_data  = datasets.FashionMNIST(root="./data", train=False, download=False, transform=transform)

BATCH_SIZE = 100
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

label_names = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]
print(f"Train: {len(train_data):,}  Test: {len(test_data):,}")

# ── Model definitions ─────────────────────────────────────────────────────────
class ModelA(nn.Module):
    """128 hidden nodes, ReLU. No regularisation, no dropout."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)

class ModelB(nn.Module):
    """48 hidden nodes, ReLU, dropout=0.2."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 48),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(48, 10),
        )
    def forward(self, x):
        return self.net(x)

# ── Training utilities ────────────────────────────────────────────────────────
def train_and_evaluate(model, optimizer, num_epochs=40):
    criterion = nn.CrossEntropyLoss()
    train_losses, test_losses = [], []
    train_accs,   test_accs   = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ── evaluate ─────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            def eval_loader(loader):
                tot_loss, correct, total = 0.0, 0, 0
                for X, y in loader:
                    X, y = X.to(device), y.to(device)
                    out  = model(X)
                    tot_loss += criterion(out, y).item()
                    correct  += (out.argmax(1) == y).sum().item()
                    total    += y.size(0)
                return tot_loss / len(loader), 100.0 * correct / total

            tr_loss, tr_acc = eval_loader(train_loader)
            te_loss, te_acc = eval_loader(test_loader)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"Train {tr_loss:.4f} / {tr_acc:.2f}% | "
                  f"Test  {te_loss:.4f} / {te_acc:.2f}%")

    return dict(
        train_losses=train_losses, test_losses=test_losses,
        train_accs=train_accs,     test_accs=test_accs,
    )

# ── Train Model A ─────────────────────────────────────────────────────────────
print("\n=== Model A: 128 nodes, no regularisation ===")
model_a  = ModelA().to(device)
opt_a    = optim.SGD(model_a.parameters(), lr=0.05)
hist_a   = train_and_evaluate(model_a, opt_a, num_epochs=40)
print(f"  Final test accuracy: {hist_a['test_accs'][-1]:.2f}%")

# ── Train Model B ─────────────────────────────────────────────────────────────
print("\n=== Model B: 48 nodes, L2 λ=1e-4, dropout 0.2 ===")
model_b  = ModelB().to(device)
opt_b    = optim.SGD(model_b.parameters(), lr=0.05, weight_decay=1e-4)
hist_b   = train_and_evaluate(model_b, opt_b, num_epochs=40)
print(f"  Final test accuracy: {hist_b['test_accs'][-1]:.2f}%")

# ═════════════════════════════════════════════════════════════════════════════
# Weight histograms
# For each model we show TWO layers: input (fc) and hidden (fc) layers.
# ═════════════════════════════════════════════════════════════════════════════

def get_weights(model):
    """Return list of (name, flat numpy weight array) for each Linear layer."""
    result = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.data.cpu().numpy().flatten()
            result.append((name, w))
    return result

weights_a = get_weights(model_a)   # [('net.1', w_input), ('net.3', w_hidden)]
weights_b = get_weights(model_b)   # [('net.1', w_input), ('net.4', w_hidden)]

layer_labels_a = ["Input layer (784→128)", "Output layer (128→10)"]
layer_labels_b = ["Input layer (784→48)",  "Output layer  (48→10)"]

# ── Figure: weight histograms ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Row 0: Model A
for col, ((name, w), lbl) in enumerate(zip(weights_a, layer_labels_a)):
    ax = axes[0, col]
    ax.hist(w, bins=60, color="steelblue", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(np.mean(w),  color="red",    linestyle="--", linewidth=1.5, label=f"mean={np.mean(w):.4f}")
    ax.axvline(np.mean(w)+np.std(w), color="orange", linestyle=":", linewidth=1.2)
    ax.axvline(np.mean(w)-np.std(w), color="orange", linestyle=":", linewidth=1.2, label=f"±1σ={np.std(w):.4f}")
    ax.set_title(f"Model A – {lbl}", fontsize=11)
    ax.set_xlabel("Weight value", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Row 1: Model B
for col, ((name, w), lbl) in enumerate(zip(weights_b, layer_labels_b)):
    ax = axes[1, col]
    ax.hist(w, bins=60, color="darkorange", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(np.mean(w),  color="red",    linestyle="--", linewidth=1.5, label=f"mean={np.mean(w):.4f}")
    ax.axvline(np.mean(w)+np.std(w), color="navy", linestyle=":", linewidth=1.2)
    ax.axvline(np.mean(w)-np.std(w), color="navy", linestyle=":", linewidth=1.2, label=f"±1σ={np.std(w):.4f}")
    ax.set_title(f"Model B – {lbl}", fontsize=11)
    ax.set_xlabel("Weight value", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Problem 2 – Weight Distributions\n"
    "Model A: 128 nodes, no regularisation  |  Model B: 48 nodes, L2 + Dropout",
    fontsize=12
)
fig.tight_layout()
fig.savefig("/home/claude/hw8/q2/p2_weight_histograms.pdf", dpi=150)
plt.close(fig)
print("\nSaved: p2_weight_histograms.pdf")

# ── Print summary statistics ──────────────────────────────────────────────────
print("\n── Weight statistics ─────────────────────────────────────────────────")
for (name, w), lbl in zip(weights_a, layer_labels_a):
    print(f"Model A  {lbl}:  std={np.std(w):.5f}  L2-norm={np.linalg.norm(w):.4f}")
for (name, w), lbl in zip(weights_b, layer_labels_b):
    print(f"Model B  {lbl}:  std={np.std(w):.5f}  L2-norm={np.linalg.norm(w):.4f}")

print("\nProblem 2 complete.")
