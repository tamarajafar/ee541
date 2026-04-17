"""
EE 541 Homework 9
Problem 1: Transfer Learning for Burning Liquid Classification

Fine-tunes a pretrained ResNet-34 on flame images of ethanol, pentane, propanol.

dataset from https://doi.org/10.1007/s10973-021-10903-2
   (Supplementary Information, File #2)=

"""

import os, re, shutil, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.utils import make_grid

from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  DATASET ORGANISATION
#     Sorts raw images (flat folder) into class subdirectories.
#     Filename pattern from the paper:  ethanol_XXXX.jpg etc.
# ══════════════════════════════════════════════════════════════════════════════
RAW_DIR  = "./raw_images"   # <-- put extracted images here
DATA_DIR = "./data"
CLASSES  = ["ethanol", "pentane", "propanol"]

def organise_dataset(raw_dir, data_dir):
    """Move/copy images from a flat folder into class sub-directories."""
    if all(os.path.isdir(os.path.join(data_dir, c)) for c in CLASSES):
        counts = {c: len(os.listdir(os.path.join(data_dir, c))) for c in CLASSES}
        total  = sum(counts.values())
        if total > 0:
            print(f"Dataset already organised: {counts}  total={total}")
            return
    os.makedirs(data_dir, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(data_dir, c), exist_ok=True)

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"\n[ERROR] Raw image directory '{raw_dir}' not found.\n"
            "Please extract the dataset there first.\n"
        )
    files = [f for f in os.listdir(raw_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    placed = 0
    for fn in files:
        fn_lower = fn.lower()
        matched  = None
        for c in CLASSES:
            if fn_lower.startswith(c):
                matched = c
                break
        if matched is None:
            # fallback: search anywhere in the filename
            for c in CLASSES:
                if c in fn_lower:
                    matched = c
                    break
        if matched:
            src = os.path.join(raw_dir, fn)
            dst = os.path.join(data_dir, matched, fn)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
            placed += 1
    counts = {c: len(os.listdir(os.path.join(data_dir, c))) for c in CLASSES}
    print(f"Organised {placed} images → {counts}")

organise_dataset(RAW_DIR, DATA_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  TRANSFORMS & DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
class_names  = full_dataset.classes          # alphabetical: ethanol/pentane/propanol
num_classes  = len(class_names)
n            = len(full_dataset)
print(f"Total images: {n}   Classes: {class_names}")

# 70 / 15 / 15 split
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
n_test  = n - n_train - n_val

indices = torch.randperm(n).tolist()
train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train + n_val]
test_idx  = indices[n_train + n_val:]

# eval transform for val & test
eval_dataset = datasets.ImageFolder(DATA_DIR, transform=eval_transform)

train_ds = Subset(full_dataset, train_idx)
val_ds   = Subset(eval_dataset, val_idx)
test_ds  = Subset(eval_dataset, test_idx)

BATCH = 32
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)
print(f"Split — train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL
# ══════════════════════════════════════════════════════════════════════════════
model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# ══════════════════════════════════════════════════════════════════════════════
# 3.  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss_sum += criterion(out, y).item()
            correct  += (out.argmax(1) == y).sum().item()
            total    += y.size(0)
    return loss_sum / len(loader), 100.0 * correct / total

def collect_preds(loader):
    """Return (all_labels, all_probs) numpy arrays."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            probs = torch.softmax(model(X), dim=1)
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  BASELINE (frozen pretrained, untrained head) accuracy
# ══════════════════════════════════════════════════════════════════════════════
_, baseline_acc = evaluate(test_loader)
print(f"\nBaseline (pretrained features, random head): {baseline_acc:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  PROGRESSIVE FINE-TUNING
#
#   Phase 1 (epochs 1-8):   freeze backbone, train fc only,   lr=1e-3
#   Phase 2 (epochs 9-16):  unfreeze layer4,                  lr=1e-4
#   Phase 3 (epochs 17-24): unfreeze layer3+layer4,           lr=5e-5
#   Phase 4 (epochs 25-30): unfreeze all,                     lr=1e-5
# ══════════════════════════════════════════════════════════════════════════════
PHASE_SCHEDULE = [
    # (start_epoch, end_epoch, unfreeze_layers, lr)
    (1,  8,  ["fc"],                          1e-3),
    (9,  16, ["fc", "layer4"],                1e-4),
    (17, 24, ["fc", "layer4", "layer3"],      5e-5),
    (25, 30, None,                            1e-5),   # None = unfreeze all
]
TOTAL_EPOCHS = PHASE_SCHEDULE[-1][1]

def set_phase(phase_start, phase_end, unfreeze_layers, lr):
    """Freeze everything, then selectively unfreeze."""
    for p in model.parameters():
        p.requires_grad = False
    if unfreeze_layers is None:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for layer_name in unfreeze_layers:
            for p in getattr(model, layer_name).parameters():
                p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=lr, weight_decay=1e-4)
    print(f"  Epochs {phase_start:2d}-{phase_end:2d} | "
          f"unfreeze={unfreeze_layers or 'ALL':30s} | "
          f"lr={lr:.0e} | trainable params={trainable:,}")
    return opt

# ── Training loop ─────────────────────────────────────────────────────────────
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
phase_markers = []         # epoch numbers where phase changes

optimizer = None
current_phase = None

print(f"\n{'='*60}")
print("Progressive fine-tuning")
print(f"{'='*60}")

for epoch in range(1, TOTAL_EPOCHS + 1):
    # ── detect phase change ──────────────────────────────────────────────────
    for (ps, pe, ul, lr) in PHASE_SCHEDULE:
        if epoch == ps:
            optimizer = set_phase(ps, pe, ul, lr)
            phase_markers.append(epoch)
            break

    # ── train one epoch ──────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    tr_loss, tr_acc = evaluate(train_loader)
    vl_loss, vl_acc = evaluate(val_loader)

    train_losses.append(tr_loss)
    val_losses.append(vl_loss)
    train_accs.append(tr_acc)
    val_accs.append(vl_acc)

    print(f"  Epoch {epoch:3d}/{TOTAL_EPOCHS} | "
          f"Train {tr_loss:.4f}/{tr_acc:.1f}% | "
          f"Val {vl_loss:.4f}/{vl_acc:.1f}%")

_, final_acc = evaluate(test_loader)
print(f"\nFinal test accuracy: {final_acc:.2f}%  (baseline was {baseline_acc:.2f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════
EPOCHS = range(1, TOTAL_EPOCHS + 1)

# ── annotation helper ──────────────────────────────────────────────────────
PHASE_LABELS = [
    "P1: FC only\n(lr=1e-3)",
    "P2: +layer4\n(lr=1e-4)",
    "P3: +layer3\n(lr=5e-5)",
    "P4: all\n(lr=1e-5)",
]
PHASE_COLORS = ["#d4e6f1", "#d5f5e3", "#fdebd0", "#f5cba7"]

def annotate_phases(ax, y_top):
    for i, (ps, pe, _, _) in enumerate(PHASE_SCHEDULE):
        ax.axvspan(ps - 0.5, pe + 0.5, alpha=0.15, color=PHASE_COLORS[i], zorder=0)
        ax.text((ps + pe) / 2, y_top, PHASE_LABELS[i],
                ha="center", va="top", fontsize=7.5, color="gray")
    for ep in phase_markers[1:]:
        ax.axvline(ep - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

# 6A. Learning curves (log-loss)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(EPOCHS, train_losses, lw=2, label="Train loss")
ax.plot(EPOCHS, val_losses,   lw=2, linestyle="--", label="Val loss")
annotate_phases(ax, ax.get_ylim()[1] if train_losses else 1)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
ax.set_title("Problem 1 – Learning Curves\n"
             "Shaded regions = training phases", fontsize=12)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("p1_learning_curves.pdf", dpi=150); plt.close(fig)
print("Saved p1_learning_curves.pdf")

# 6B. Accuracy curves
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(EPOCHS, train_accs, lw=2, label="Train accuracy")
ax.plot(EPOCHS, val_accs,   lw=2, linestyle="--", label="Val accuracy")
annotate_phases(ax, max(max(train_accs), max(val_accs)) + 1)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Problem 1 – Accuracy Curves", fontsize=12)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("p1_accuracy_curves.pdf", dpi=150); plt.close(fig)
print("Saved p1_accuracy_curves.pdf")

# 6C. Confusion matrix
test_labels, test_probs = collect_preds(test_loader)
test_preds = test_probs.argmax(axis=1)

cm      = confusion_matrix(test_labels, test_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor="gray", ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title(f"Problem 1 – Confusion Matrix (row-normalised)\n"
             f"Test accuracy: {final_acc:.2f}%  |  Baseline: {baseline_acc:.2f}%",
             fontsize=12)
fig.tight_layout()
fig.savefig("p1_confusion_matrix.pdf", dpi=150); plt.close(fig)
print("Saved p1_confusion_matrix.pdf")

# 6D. Precision-Recall curves (one-vs-rest)
y_bin   = label_binarize(test_labels, classes=list(range(num_classes)))
COLORS  = ["steelblue", "darkorange", "forestgreen"]

fig, ax = plt.subplots(figsize=(7, 5.5))
for i, (cname, color) in enumerate(zip(class_names, COLORS)):
    prec, rec, _ = precision_recall_curve(y_bin[:, i], test_probs[:, i])
    ap = average_precision_score(y_bin[:, i], test_probs[:, i])
    ax.plot(rec, prec, lw=2, color=color, label=f"{cname}  AP={ap:.3f}")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Problem 1 – Precision-Recall Curves (one-vs-rest)", fontsize=12)
ax.legend(fontsize=11, loc="lower left")
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
fig.tight_layout()
fig.savefig("p1_precision_recall.pdf", dpi=150); plt.close(fig)
print("Saved p1_precision_recall.pdf")

# 6E. Baseline vs Fine-tuned bar chart
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(["Pretrained\n(frozen, random head)", "Fine-tuned\nResNet-34"],
              [baseline_acc, final_acc],
              color=["#aed6f1", "#1a5276"], width=0.45, edgecolor="white")
for bar, val in zip(bars, [baseline_acc, final_acc]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5, f"{val:.2f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("Problem 1 – Baseline vs Fine-tuned Model", fontsize=12)
ax.set_ylim([0, 105]); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("p1_baseline_vs_finetuned.pdf", dpi=150); plt.close(fig)
print("Saved p1_baseline_vs_finetuned.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  FEATURE MAP VISUALISATION (forward hooks)
# ══════════════════════════════════════════════════════════════════════════════

# Grab one real test image
sample_img_tensor = None
for X, _ in test_loader:
    sample_img_tensor = X[0:1].to(device)   # shape [1,3,224,224]
    break

def save_feature_maps(layer, layer_name, sample_tensor, filename,
                      max_filters=64, title_prefix=""):
    """Register a hook, run one forward pass, save the feature map grid."""
    captured = {}

    def hook_fn(module, inp, out):
        captured["out"] = out.detach().cpu()

    h = layer.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        _ = model(sample_tensor)
    h.remove()

    fmaps = captured["out"][0]           # (C, H, W)
    n_filters = min(fmaps.shape[0], max_filters)
    fmaps = fmaps[:n_filters]

    # normalise each channel independently to [0,1]
    vmin = fmaps.flatten(1).min(dim=1).values[:, None, None]
    vmax = fmaps.flatten(1).max(dim=1).values[:, None, None]
    fmaps = (fmaps - vmin) / (vmax - vmin + 1e-8)

    ncols = 8
    nrows = (n_filters + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.6, nrows * 1.6))
    axes = np.array(axes).flatten()
    for i in range(nrows * ncols):
        ax = axes[i]
        if i < n_filters:
            ax.imshow(fmaps[i].numpy(), cmap="viridis")
            ax.set_title(f"f{i}", fontsize=6)
        ax.axis("off")
    fig.suptitle(f"{title_prefix}{layer_name}  ({fmaps.shape[0]} filters shown)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"Saved {filename}")

# First conv layer  (64 filters of 7×7, output 64×112×112)
save_feature_maps(model.conv1,
                  layer_name="conv1",
                  sample_tensor=sample_img_tensor,
                  filename="p1_feature_maps_conv1.pdf",
                  max_filters=64,
                  title_prefix="Feature maps – ")

# Mid-network: first conv in layer2 (128 filters)
save_feature_maps(model.layer2[0].conv1,
                  layer_name="layer2[0].conv1",
                  sample_tensor=sample_img_tensor,
                  filename="p1_feature_maps_mid.pdf",
                  max_filters=64,
                  title_prefix="Feature maps – ")

# Deep layer: first conv in layer4 (512 filters, show first 64)
save_feature_maps(model.layer4[0].conv1,
                  layer_name="layer4[0].conv1",
                  sample_tensor=sample_img_tensor,
                  filename="p1_feature_maps_deep.pdf",
                  max_filters=64,
                  title_prefix="Feature maps – ")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.metrics import classification_report
print("\n── Classification Report (test set) ─────────────────────────────────")
print(classification_report(test_labels, test_preds, target_names=class_names))

print("\n── Per-class Average Precision ───────────────────────────────────────")
for i, cname in enumerate(class_names):
    ap = average_precision_score(y_bin[:, i], test_probs[:, i])
    print(f"  {cname:10s}  AP = {ap:.4f}")

print(f"\nBaseline accuracy : {baseline_acc:.2f}%")
print(f"Fine-tuned accuracy: {final_acc:.2f}%")
print(f"Improvement       : +{final_acc - baseline_acc:.2f} pp")

print("\nAll output files saved. Problem 1 complete.")
