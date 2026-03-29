"""
EE 541 – Homework 7, Problem 1
MLP Training with NumPy (no PyTorch / tf.keras / scikit-learn)

Architecture  : 784 → 200 → 100 → 10  (matches mnist_network_params.hdf5)
Data file     : mnist_traindata.hdf5
  xdata: (60000, 784) float32, already normalized [0,1]
  ydata: (60000, 10)  float64, one-hot encoded
Params file   : mnist_network_params.hdf5  (warm-start for final model)

Activations   : tanh and ReLU (both swept)
Loss          : Cross-entropy
Optimizer     : Mini-batch SGD, batch_size = 100
Initialisation: He (ReLU) / Xavier (tanh) for the 6-config sweep
                Provided params file used as warm-start for the final model
LR decay      : divide by 2 after epochs 20 and 40
Regularisation: L2, lambda = 1e-4
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import os

# ── Paths ──────────────────────────────────────────────────────
TRAIN_FILE  = "mnist_traindata.hdf5"
PARAMS_FILE = "mnist_network_params.hdf5"
# TEST_FILE = "mnist_testdata.hdf5"    # uncomment when available

# ── Hyper-parameters ───────────────────────────────────────────
EPOCHS          = 50
BATCH_SIZE      = 100            # 500 updates/epoch on 50k training set
LR_DECAY_EPOCHS = [20, 40]       # divide LR by 2 after these epochs
LAMBDA_L2       = 1e-4

LAYER_SIZES    = [784, 200, 100, 10]   # matches the provided params file
LEARNING_RATES = [0.1, 0.01, 0.001]


# ══════════════════════════════════════════════════════════════
# Activation functions and their derivatives
# ══════════════════════════════════════════════════════════════
def relu(x):
    return np.maximum(0.0, x)

def relu_deriv(x):
    """Derivative of ReLU. Value at x=0 set to 0 (valid: any value in [0,1])."""
    return (x > 0).astype(float)

def tanh_act(x):
    return np.tanh(x)

def tanh_deriv(x):
    """Derivative of tanh: 1 - tanh^2(x)"""
    t = np.tanh(x)
    return 1.0 - t * t


# ══════════════════════════════════════════════════════════════
# Numerically stable softmax and cross-entropy
# ══════════════════════════════════════════════════════════════
def softmax(z):
    """Numerically stable softmax. z: (10, batch)"""
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z_shifted)
    return e / np.sum(e, axis=0, keepdims=True)

def cross_entropy_loss(a_out, y_oh):
    """
    Mean cross-entropy loss.
    a_out : (10, batch) – softmax probabilities
    y_oh  : (10, batch) – one-hot targets
    """
    eps = 1e-15
    return -np.mean(np.sum(y_oh * np.log(np.clip(a_out, eps, 1.0)), axis=0))


# ══════════════════════════════════════════════════════════════
# Weight initialisation
# ══════════════════════════════════════════════════════════════
def init_params_random(layer_sizes, activation):
    """
    He initialisation for ReLU, Xavier/Glorot for tanh.
    Returns list of mutable [W, b] pairs.
    """
    params = []
    for i in range(len(layer_sizes) - 1):
        fan_in  = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        if activation == "relu":
            std = np.sqrt(2.0 / fan_in)               # He
        else:
            std = np.sqrt(2.0 / (fan_in + fan_out))   # Xavier
        W = np.random.randn(fan_out, fan_in) * std
        b = np.zeros((fan_out, 1))
        params.append([W, b])
    return params

def init_params_from_file(filepath):
    """
    Load pre-trained weights from mnist_network_params.hdf5.
    Architecture: 784 -> 200 -> 100 -> 10.
    """
    with h5py.File(filepath, "r") as f:
        params = [
            [f["W1"][()].astype(np.float64), f["b1"][()].astype(np.float64).reshape(-1, 1)],
            [f["W2"][()].astype(np.float64), f["b2"][()].astype(np.float64).reshape(-1, 1)],
            [f["W3"][()].astype(np.float64), f["b3"][()].astype(np.float64).reshape(-1, 1)],
        ]
    return params


# ══════════════════════════════════════════════════════════════
# Forward pass
# ══════════════════════════════════════════════════════════════
def forward(X, params, act_fn):
    """
    Forward propagation through the MLP.

    Equations (per layer l):
      s^(l) = W^(l) a^(l-1) + b^(l)
      a^(l) = h(s^(l))              [hidden layers]
      a^(L) = softmax(s^(L))        [output layer]

    X      : (784, batch)
    act_fn : 'relu' or 'tanh'
    Returns (y_hat, cache)
      cache = [(a_prev, s, a), ...] one entry per layer
    """
    cache = []
    a = X
    hidden_fn = relu if act_fn == "relu" else tanh_act

    for W, b in params[:-1]:       # hidden layers
        a_prev = a
        s = W @ a_prev + b
        a = hidden_fn(s)
        cache.append((a_prev, s, a))

    # output layer -> softmax
    W, b = params[-1]
    a_prev = a
    s = W @ a_prev + b
    a_out = softmax(s)
    cache.append((a_prev, s, a_out))
    return a_out, cache


# ══════════════════════════════════════════════════════════════
# Backward pass  (delta method)
# ══════════════════════════════════════════════════════════════
def backward(y_oh, cache, params, act_fn, lambda_l2):
    """
    Backpropagation using the delta method.

    Output layer:   delta^(L) = a^(L) - y
                    (combined softmax + cross-entropy gradient)
    Hidden layers:  delta^(l) = (W^(l+1)^T delta^(l+1)) * h'(s^(l))
    Weight grad:    dC/dW^(l) = delta^(l)(a^(l-1))^T / m  +  lambda * W^(l)
    Bias grad:      dC/db^(l) = mean(delta^(l), axis=1)

    Returns list of (dW, db) per layer.
    """
    m = y_oh.shape[1]
    grads = [None] * len(params)
    hidden_deriv = relu_deriv if act_fn == "relu" else tanh_deriv

    # Output layer
    a_out  = cache[-1][2]
    delta  = a_out - y_oh
    a_prev = cache[-1][0]
    dW = (delta @ a_prev.T) / m  +  lambda_l2 * params[-1][0]
    db = np.mean(delta, axis=1, keepdims=True)
    grads[-1] = (dW, db)

    # Hidden layers in reverse
    for l in range(len(params) - 2, -1, -1):
        W_next = params[l + 1][0]
        delta  = (W_next.T @ delta) * hidden_deriv(cache[l][1])
        a_prev = cache[l][0]
        dW = (delta @ a_prev.T) / m  +  lambda_l2 * params[l][0]
        db = np.mean(delta, axis=1, keepdims=True)
        grads[l] = (dW, db)

    return grads


# ══════════════════════════════════════════════════════════════
# Accuracy helper
# ══════════════════════════════════════════════════════════════
def accuracy(X, y_labels, params, act_fn):
    """y_labels: integer class indices, shape (n,)"""
    a_out, _ = forward(X, params, act_fn)
    preds = np.argmax(a_out, axis=0)
    return np.mean(preds == y_labels)


# ══════════════════════════════════════════════════════════════
# One full training run
# ══════════════════════════════════════════════════════════════
def train(X_train, y_train_oh, y_train_lbl,
          X_val,   y_val_lbl,
          act_fn, lr_init, init_params_fn,
          epochs=EPOCHS, batch_size=BATCH_SIZE,
          lambda_l2=LAMBDA_L2, lr_decay_epochs=LR_DECAY_EPOCHS,
          seed=42, verbose=True):
    """
    Train for one configuration.
    - Gradients averaged over each mini-batch.
    - LR divided by 2 at the start of the epoch after each lr_decay_epoch.
    """
    np.random.seed(seed)
    params = init_params_fn()
    lr = lr_init
    n_train = X_train.shape[1]
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(1, epochs + 1):
        if (epoch - 1) in lr_decay_epochs:
            lr /= 2.0

        perm   = np.random.permutation(n_train)
        X_shuf = X_train[:, perm]
        y_shuf = y_train_oh[:, perm]

        for start in range(0, n_train, batch_size):
            Xb = X_shuf[:, start:start + batch_size]
            Yb = y_shuf[:, start:start + batch_size]
            a_out, cache = forward(Xb, params, act_fn)
            grads = backward(Yb, cache, params, act_fn, lambda_l2)
            for l in range(len(params)):
                params[l][0] -= lr * grads[l][0]
                params[l][1] -= lr * grads[l][1]

        tr_acc  = accuracy(X_train, y_train_lbl, params, act_fn)
        val_acc = accuracy(X_val,   y_val_lbl,   params, act_fn)
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(val_acc)

        if verbose:
            print(f"  Epoch {epoch:3d} | lr={lr:.6f} | "
                  f"train={tr_acc*100:.2f}%  val={val_acc*100:.2f}%")

    return params, train_acc_hist, val_acc_hist


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Load data ──────────────────────────────────────────────
    print("Loading mnist_traindata.hdf5 ...")
    with h5py.File(TRAIN_FILE, "r") as f:
        X_all = f["xdata"][()].astype(np.float64).T   # (784, 60000)  already [0,1]
        Y_oh  = f["ydata"][()].astype(np.float64).T   # (10,  60000)  one-hot
    y_all = np.argmax(Y_oh, axis=0)                    # integer labels (60000,)

    X_train     = X_all[:, :50000]    # (784, 50000)
    y_train_oh  = Y_oh[:,  :50000]    # (10,  50000)
    y_train_lbl = y_all[:50000]
    X_val       = X_all[:, 50000:]    # (784, 10000)
    y_val_lbl   = y_all[50000:]
    print(f"  Train: {X_train.shape[1]}   Val: {X_val.shape[1]}")

    # ── 6-config sweep ─────────────────────────────────────────
    configs = [(act, lr)
               for act in ["tanh", "relu"]
               for lr  in LEARNING_RATES]

    best_val_acc   = -1.0
    best_cfg       = None
    all_train_hist = {}
    all_val_hist   = {}

    for act_fn, lr_init in configs:
        run_id = f"{act_fn}_lr{lr_init}"
        print(f"\n{'='*60}")
        print(f"  activation={act_fn}   initial_lr={lr_init}")
        print(f"{'='*60}")

        init_fn = lambda a=act_fn: init_params_random(LAYER_SIZES, a)
        params, tr_hist, val_hist = train(
            X_train, y_train_oh, y_train_lbl,
            X_val,   y_val_lbl,
            act_fn=act_fn, lr_init=lr_init,
            init_params_fn=init_fn
        )
        all_train_hist[run_id] = tr_hist
        all_val_hist[run_id]   = val_hist

        best_this = max(val_hist)
        print(f"  -> Best val accuracy: {best_this*100:.2f}%")

        # Individual curve PDF
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ep = range(1, EPOCHS + 1)
        ax.plot(ep, [v*100 for v in tr_hist],  label="Train Accuracy")
        ax.plot(ep, [v*100 for v in val_hist], label="Val  Accuracy")
        for dep in LR_DECAY_EPOCHS:
            ax.axvline(x=dep, color="gray", linestyle="--", alpha=0.7,
                       label=f"LR/2 @ epoch {dep}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Activation={act_fn.upper()}  |  Initial LR={lr_init}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"curve_{run_id}.pdf"))
        plt.close(fig)
        print(f"  Saved curve_{run_id}.pdf")

        if best_this > best_val_acc:
            best_val_acc = best_this
            best_cfg     = (act_fn, lr_init)

    print(f"\n{'='*60}")
    print(f"  Best config: activation={best_cfg[0]}  lr={best_cfg[1]}")
    print(f"  Best validation accuracy = {best_val_acc*100:.2f}%")

    # ── 2x3 summary figure ─────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx, (act_fn, lr_init) in enumerate(configs):
        run_id = f"{act_fn}_lr{lr_init}"
        ax = axes.flatten()[idx]
        ep = range(1, EPOCHS + 1)
        ax.plot(ep, [v*100 for v in all_train_hist[run_id]], label="Train")
        ax.plot(ep, [v*100 for v in all_val_hist[run_id]],   label="Val")
        for dep in LR_DECAY_EPOCHS:
            ax.axvline(x=dep, color="gray", linestyle="--", alpha=0.6)
        star = "* " if (act_fn, lr_init) == best_cfg else ""
        ax.set_title(f"{star}{act_fn.upper()} | LR={lr_init}", fontsize=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"6-Config Sweep  |  Network: {LAYER_SIZES}  |  "
        f"Batch={BATCH_SIZE}  |  L2={LAMBDA_L2}\n"
        f"Best: {best_cfg[0].upper()} LR={best_cfg[1]}  "
        f"Val={best_val_acc*100:.2f}%",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "all_curves_summary.pdf"))
    plt.close(fig)
    print("Saved all_curves_summary.pdf")

    # ── Final model on all 60,000 images ───────────────────────
    print(f"\n{'='*60}")
    print("  Final model: all 60,000 images")
    print(f"  Warm-start from {PARAMS_FILE}")
    print(f"{'='*60}")

    best_act, best_lr = best_cfg
    final_params, _, _ = train(
        X_all, Y_oh, y_all,
        X_all, y_all,           # track train accuracy (no held-out val)
        act_fn=best_act, lr_init=best_lr,
        init_params_fn=lambda: init_params_from_file(PARAMS_FILE),
        verbose=True
    )

    # ── Test set (uncomment when mnist_testdata.hdf5 is available) ──
    # with h5py.File(TEST_FILE, "r") as f:
    #     X_test = f["xdata"][()].astype(np.float64).T
    #     y_test = np.argmax(f["ydata"][()].astype(np.float64).T, axis=0)
    # test_acc = accuracy(X_test, y_test, final_params, best_act)
    # print(f"\n  FINAL TEST ACCURACY = {test_acc*100:.2f}%")

    # ── Print submission summary ────────────────────────────────
    print(f"\n{'='*60}")
    print("SUBMISSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Network        : {LAYER_SIZES}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Learning rates : {LEARNING_RATES}")
    print(f"  LR decay       : divided by 2 after epochs {LR_DECAY_EPOCHS}")
    print(f"  Activations    : tanh (Xavier init)  &  ReLU (He init)")
    print(f"  Regularisation : L2, lambda = {LAMBDA_L2}")
    print(f"  Best config    : act={best_cfg[0]}  lr={best_cfg[1]}")
    print(f"  Best val acc   : {best_val_acc*100:.2f}%")
    print(f"  Final model    : warm-started from {PARAMS_FILE}")


if __name__ == "__main__":
    main()
