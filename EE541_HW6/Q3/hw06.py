"""
EE 541 - Homework #6, Problem 3
Softmax / Multi-class Logistic Regression on MNIST
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_all = mnist.data.astype(np.float64) / 255.0
y_all = mnist.target.astype(np.int32)

X_train = X_all[:60000]
y_train = y_all[:60000]
X_test  = X_all[60000:]
y_test  = y_all[60000:]

N_train, D = X_train.shape
N_test      = X_test.shape[0]
K           = 10

print(f"Train: {N_train} samples, {D} features")
print(f"Test : {N_test}  samples")

# Helper functions
def softmax(scores):
    s = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)

def forward(X, W, b):
    return softmax(X @ W.T + b)

def cross_entropy_loss(probs, y):
    N = len(y)
    log_p = np.log(probs[np.arange(N), y] + 1e-15)
    return -log_p.mean()

def accuracy(probs, y):
    return (probs.argmax(axis=1) == y).mean()

def gradient(X, probs, y):
    N = len(y)
    delta = probs.copy()
    delta[np.arange(N), y] -= 1.0
    dW = (delta.T @ X) / N
    db = delta.mean(axis=0)
    return dW, db

def init_weights(K, D, seed=42):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((K, D)) * 0.01
    b = np.zeros(K)
    return W, b

# batch gradient descent 
print("\n" + "="*60)
print("PART (b): BATCH GRADIENT DESCENT")
print("="*60)

LEARNING_RATE_BGD = 0.5
N_ITER_BGD        = 300

W_bgd, b_bgd = init_weights(K, D)

train_loss_bgd = []
test_loss_bgd  = []
train_acc_bgd  = []
test_acc_bgd   = []

for it in range(N_ITER_BGD):
    probs_tr = forward(X_train, W_bgd, b_bgd)
    probs_te = forward(X_test,  W_bgd, b_bgd)

    trl = cross_entropy_loss(probs_tr, y_train)
    tel = cross_entropy_loss(probs_te, y_test)
    tra = accuracy(probs_tr, y_train)
    tea = accuracy(probs_te, y_test)

    train_loss_bgd.append(trl)
    test_loss_bgd.append(tel)
    train_acc_bgd.append(tra)
    test_acc_bgd.append(tea)

    dW, db = gradient(X_train, probs_tr, y_train)
    W_bgd -= LEARNING_RATE_BGD * dW
    b_bgd -= LEARNING_RATE_BGD * db

    if it % 50 == 0:
        print(f"  Iter {it:4d}: train_loss={trl:.4f}  test_loss={tel:.4f}"
              f"  train_acc={tra:.4f}  test_acc={tea:.4f}")

print(f"\nFinal BGD results:")
print(f"  Train loss={train_loss_bgd[-1]:.4f}  acc={train_acc_bgd[-1]:.4f}")
print(f"  Test  loss={test_loss_bgd[-1]:.4f}  acc={test_acc_bgd[-1]:.4f}")
print(f"  Learning rate: {LEARNING_RATE_BGD}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
iters = np.arange(N_ITER_BGD)
ax1.plot(iters, train_loss_bgd, label='Train loss')
ax1.plot(iters, test_loss_bgd,  label='Test loss', linestyle='--')
ax1.set_xlabel('Iteration'); ax1.set_ylabel('Log-loss')
ax1.set_title('BGD - Learning Curves')
ax1.legend(); ax1.grid(True)
ax2.plot(iters, train_acc_bgd, label='Train accuracy')
ax2.plot(iters, test_acc_bgd,  label='Test accuracy', linestyle='--')
ax2.set_xlabel('Iteration'); ax2.set_ylabel('Accuracy')
ax2.set_title('BGD - Accuracy Curves')
ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig('hw06_bgd_curves.png', dpi=150)
plt.close()
print("  Saved: hw06_bgd_curves.png")

# sgd bacth size 1
print("\n" + "="*60)
print("PART (c-1): SGD, MINI-BATCH SIZE = 1")
print("="*60)

LEARNING_RATE_SGD1 = 0.01
N_EPOCHS_SGD1      = 3
RECORD_EVERY       = 5000

W_sgd1, b_sgd1 = init_weights(K, D)

sgd1_train_loss  = []
sgd1_test_loss   = []
sgd1_train_acc   = []
sgd1_test_acc    = []
sgd1_sample_nums = []

sample_count = 0

probs_tr = forward(X_train, W_sgd1, b_sgd1)
probs_te = forward(X_test,  W_sgd1, b_sgd1)
sgd1_train_loss.append(cross_entropy_loss(probs_tr, y_train))
sgd1_test_loss.append(cross_entropy_loss(probs_te, y_test))
sgd1_train_acc.append(accuracy(probs_tr, y_train))
sgd1_test_acc.append(accuracy(probs_te, y_test))
sgd1_sample_nums.append(0)

for epoch in range(N_EPOCHS_SGD1):
    idx    = np.random.permutation(N_train)
    X_shuf = X_train[idx]
    y_shuf = y_train[idx]

    for i in range(N_train):
        xi = X_shuf[i:i+1]
        yi = y_shuf[i:i+1]

        probs_i = forward(xi, W_sgd1, b_sgd1)
        dW, db  = gradient(xi, probs_i, yi)
        W_sgd1 -= LEARNING_RATE_SGD1 * dW
        b_sgd1 -= LEARNING_RATE_SGD1 * db

        sample_count += 1
        if sample_count % RECORD_EVERY == 0:
            probs_tr = forward(X_train, W_sgd1, b_sgd1)
            probs_te = forward(X_test,  W_sgd1, b_sgd1)
            sgd1_train_loss.append(cross_entropy_loss(probs_tr, y_train))
            sgd1_test_loss.append(cross_entropy_loss(probs_te, y_test))
            sgd1_train_acc.append(accuracy(probs_tr, y_train))
            sgd1_test_acc.append(accuracy(probs_te, y_test))
            sgd1_sample_nums.append(sample_count)

    print(f"  Epoch {epoch+1}: train_loss={sgd1_train_loss[-1]:.4f}  "
          f"test_loss={sgd1_test_loss[-1]:.4f}  "
          f"train_acc={sgd1_train_acc[-1]:.4f}  "
          f"test_acc={sgd1_test_acc[-1]:.4f}")

print(f"\nFinal SGD-1 results:")
print(f"  Train loss={sgd1_train_loss[-1]:.4f}  acc={sgd1_train_acc[-1]:.4f}")
print(f"  Test  loss={sgd1_test_loss[-1]:.4f}  acc={sgd1_test_acc[-1]:.4f}")
print(f"  Learning rate: {LEARNING_RATE_SGD1}")

# sgd batch size 100
print("\n" + "="*60)
print("PART (c-3): SGD, MINI-BATCH SIZE = 100")
print("="*60)

LEARNING_RATE_SGD100 = 0.1
BATCH_SIZE           = 100
N_EPOCHS_SGD100      = 10

W_sgd100, b_sgd100 = init_weights(K, D)

sgd100_train_loss  = []
sgd100_test_loss   = []
sgd100_train_acc   = []
sgd100_test_acc    = []
sgd100_sample_nums = []

sample_count = 0

probs_tr = forward(X_train, W_sgd100, b_sgd100)
probs_te = forward(X_test,  W_sgd100, b_sgd100)
sgd100_train_loss.append(cross_entropy_loss(probs_tr, y_train))
sgd100_test_loss.append(cross_entropy_loss(probs_te, y_test))
sgd100_train_acc.append(accuracy(probs_tr, y_train))
sgd100_test_acc.append(accuracy(probs_te, y_test))
sgd100_sample_nums.append(0)

for epoch in range(N_EPOCHS_SGD100):
    idx      = np.random.permutation(N_train)
    X_shuf   = X_train[idx]
    y_shuf   = y_train[idx]
    n_batches = N_train // BATCH_SIZE

    for batch_idx in range(n_batches):
        Xb = X_shuf[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
        yb = y_shuf[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]

        probs_b = forward(Xb, W_sgd100, b_sgd100)
        dW, db  = gradient(Xb, probs_b, yb)
        W_sgd100 -= LEARNING_RATE_SGD100 * dW
        b_sgd100 -= LEARNING_RATE_SGD100 * db

        sample_count += BATCH_SIZE
        if sample_count % RECORD_EVERY == 0:
            probs_tr = forward(X_train, W_sgd100, b_sgd100)
            probs_te = forward(X_test,  W_sgd100, b_sgd100)
            sgd100_train_loss.append(cross_entropy_loss(probs_tr, y_train))
            sgd100_test_loss.append(cross_entropy_loss(probs_te, y_test))
            sgd100_train_acc.append(accuracy(probs_tr, y_train))
            sgd100_test_acc.append(accuracy(probs_te, y_test))
            sgd100_sample_nums.append(sample_count)

    print(f"  Epoch {epoch+1}: train_loss={sgd100_train_loss[-1]:.4f}  "
          f"test_loss={sgd100_test_loss[-1]:.4f}  "
          f"train_acc={sgd100_train_acc[-1]:.4f}  "
          f"test_acc={sgd100_test_acc[-1]:.4f}")

print(f"\nFinal SGD-100 results:")
print(f"  Train loss={sgd100_train_loss[-1]:.4f}  acc={sgd100_train_acc[-1]:.4f}")
print(f"  Test  loss={sgd100_test_loss[-1]:.4f}  acc={sgd100_test_acc[-1]:.4f}")
print(f"  Learning rate: {LEARNING_RATE_SGD100}")

#  SGD plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(sgd1_sample_nums, sgd1_train_loss, label='Train')
axes[0,0].plot(sgd1_sample_nums, sgd1_test_loss,  label='Test', linestyle='--')
axes[0,0].set_xlabel('Samples seen'); axes[0,0].set_ylabel('Log-loss')
axes[0,0].set_title('SGD (batch=1) - Learning Curves')
axes[0,0].legend(); axes[0,0].grid(True)

axes[0,1].plot(sgd1_sample_nums, sgd1_train_acc, label='Train')
axes[0,1].plot(sgd1_sample_nums, sgd1_test_acc,  label='Test', linestyle='--')
axes[0,1].set_xlabel('Samples seen'); axes[0,1].set_ylabel('Accuracy')
axes[0,1].set_title('SGD (batch=1) - Accuracy Curves')
axes[0,1].legend(); axes[0,1].grid(True)

axes[1,0].plot(sgd100_sample_nums, sgd100_train_loss, label='Train')
axes[1,0].plot(sgd100_sample_nums, sgd100_test_loss,  label='Test', linestyle='--')
axes[1,0].set_xlabel('Samples seen'); axes[1,0].set_ylabel('Log-loss')
axes[1,0].set_title('SGD (batch=100) - Learning Curves')
axes[1,0].legend(); axes[1,0].grid(True)

axes[1,1].plot(sgd100_sample_nums, sgd100_train_acc, label='Train')
axes[1,1].plot(sgd100_sample_nums, sgd100_test_acc,  label='Test', linestyle='--')
axes[1,1].set_xlabel('Samples seen'); axes[1,1].set_ylabel('Accuracy')
axes[1,1].set_title('SGD (batch=100) - Accuracy Curves')
axes[1,1].legend(); axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('hw06_sgd_curves.png', dpi=150)
plt.close()
print("\nSaved: hw06_sgd_curves.png")

import h5py
with h5py.File('weights.h5', 'w') as f:
    f.create_dataset('W', data=W_sgd100)
    f.create_dataset('b', data=b_sgd100)
print("\nSaved: weights.h5")


print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"{'Method':<20} {'Train Loss':>12} {'Test Loss':>12} {'Train Acc':>12} {'Test Acc':>12}")
print("-"*70)
print(f"{'BGD (eta=0.5)':<20} {train_loss_bgd[-1]:>12.4f} {test_loss_bgd[-1]:>12.4f} "
      f"{train_acc_bgd[-1]:>12.4f} {test_acc_bgd[-1]:>12.4f}")
print(f"{'SGD-1 (eta=0.01)':<20} {sgd1_train_loss[-1]:>12.4f} {sgd1_test_loss[-1]:>12.4f} "
      f"{sgd1_train_acc[-1]:>12.4f} {sgd1_test_acc[-1]:>12.4f}")
print(f"{'SGD-100 (eta=0.1)':<20} {sgd100_train_loss[-1]:>12.4f} {sgd100_test_loss[-1]:>12.4f} "
      f"{sgd100_train_acc[-1]:>12.4f} {sgd100_test_acc[-1]:>12.4f}")