#!/usr/bin/env python
# coding: utf-8

# In[12]:


import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


# In[20]:


CLASS_MAP = {'Head': 0, 'Ear_Left': 1, 'Ear_Right': 2}

pattern = re.compile(
    r'^\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)'   # x number
    r'\s+'                                            # SPACE
    r'([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)'         # y number
    r'\s+'                                            # SPACE
    r'(Head|Ear_Left|Ear_Right)'                      # class name
    r'\s*$'
)

X_list = []
y_list = []

with open('/Users/tamarajafar/Downloads/cluster.txt', 'r') as f:
    for line in f:
        if line.startswith('#'):   # skip comment lines
            continue
        m = pattern.match(line)
        if m:
            X_list.append([float(m.group(1)), float(m.group(2))])
            y_list.append(CLASS_MAP[m.group(3)])

X      = np.array(X_list)
y_true = np.array(y_list)

print(f"Loaded {len(X)} points")
print(f"X shape: {X.shape}")


# In[21]:


# skip comment lines and show the first 5 actual data lines
with open('/Users/tamarajafar/Downloads/cluster.txt', 'r') as f:
    count = 0
    for line in f:
        if not line.startswith('#'):
            print(repr(line))
            count += 1
            if count >= 5:
                break


# In[37]:


#kmeans 
km = KMeans(n_clusters=3, n_init=20, random_state=0)
km.fit(X)
raw_labels = km.labels_
print(raw_labels[:10]) 

# Remap by majority-vote against ground truth
remap = {}
for k in range(3):
    mask = raw_labels == k
    counts = np.bincount(y_true[mask], minlength=3)
    remap[k] = np.argmax(counts)
km_labels = np.array([remap[raw] for raw in raw_labels])

cm_km  = confusion_matrix(y_true, km_labels, labels=[0, 1, 2])
acc_km = np.trace(cm_km) / cm_km.sum()
print(f"K-Means Accuracy: {acc_km:.1%}")


# In[39]:


# Scatter
fig, ax = plt.subplots(figsize=(7, 6))
for cls in range(3):
    mask = km_labels == cls
    ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[cls], label=CLASS_NAMES[cls], s=30, alpha=0.8)
ax.set_title('K-Means (K=3) Cluster Assignments')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend()
plt.tight_layout()
plt.savefig('/Users/tamarajafar/Desktop/EE541/EE541_HW4/Q1/fig_kmeans_scatter.png', dpi=150)
plt.close()


# In[41]:


# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm_km, cmap='Blues')
ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'K-Means Confusion Matrix (acc={acc_km:.1%})')
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm_km[i, j], ha='center', va='center', fontsize=12,
                color='white' if cm_km[i, j] > cm_km.max() / 2 else 'black')
plt.tight_layout()
plt.savefig('/Users/tamarajafar/Desktop/EE541/EE541_HW4/Q1/fig_kmeans_confusion.png', dpi=150)
plt.close()
print("K-Means figures saved")


# In[42]:


#GMM HELPERS 

def gaussian_pdf(X, mu, sigma):
    N, D    = X.shape
    sig_reg = sigma + 1e-6 * np.eye(D)
    diff    = X - mu
    inv_sig = np.linalg.inv(sig_reg)
    maha    = np.einsum('ni,ij,nj->n', diff, inv_sig, diff)
    log_det = np.log(np.linalg.det(sig_reg) + 1e-300)
    log_pdf = -0.5 * (D * np.log(2 * np.pi) + log_det + maha)
    return np.exp(log_pdf)

def compute_nll(X, pi, mus, sigmas):
    likelihood = np.zeros(len(X))
    for k in range(3):
        likelihood += pi[k] * gaussian_pdf(X, mus[k], sigmas[k])
    return -np.sum(np.log(likelihood + 1e-300))


# In[27]:


N, D = X.shape
K    = 3

gamma = np.zeros((N, K))
for n in range(N):
    gamma[n, km_labels[n]] = 1.0

print("First 5 rows of gamma:")
print(gamma[:5])

# Compute starting parameters from this gamma
N_k    = gamma.sum(axis=0)
pi     = N_k / N
mus    = (gamma.T @ X) / N_k[:, None]
sigmas = []
for k in range(K):
    diff  = X - mus[k]
    sigma = (gamma[:, k:k+1] * diff).T @ diff / N_k[k]
    sigmas.append(sigma)

print("\nInitial means:")
for k, mu in enumerate(mus):
    print(f"  Component {k}: x={mu[0]:.3f}, y={mu[1]:.3f}")


# In[43]:


# GM/EM
N, D, K = X.shape[0], X.shape[1], 3

# Initialize with soft gamma (0.95/0.025) to avoid degenerate covariances
gamma = np.full((N, K), 0.025)
for n in range(N):
    gamma[n, km_labels[n]] = 0.95

N_k    = gamma.sum(axis=0)
pi     = N_k / N
mus    = (gamma.T @ X) / N_k[:, None]
sigmas = []
for k in range(K):
    diff  = X - mus[k]
    sigma = (gamma[:, k:k+1] * diff).T @ diff / N_k[k]
    sigmas.append(sigma)

nll_history = [compute_nll(X, pi, mus, sigmas)]
snapshots   = []
print(f"Starting NLL: {nll_history[0]:.4f}")

for iteration in range(1, 201):
    # E-step
    r = np.zeros((N, K))
    for k in range(K):
        r[:, k] = pi[k] * gaussian_pdf(X, mus[k], sigmas[k])
    row_sums = r.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-300, 1e-300, row_sums)
    gamma    = r / row_sums

    if iteration <= 4:
        snapshots.append((iteration, np.argmax(gamma, axis=1).copy()))

    # M-step
    N_k    = gamma.sum(axis=0)
    pi     = N_k / N
    mus    = (gamma.T @ X) / N_k[:, None]
    sigmas = []
    for k in range(K):
        diff  = X - mus[k]
        sigma = (gamma[:, k:k+1] * diff).T @ diff / N_k[k]
        sigmas.append(sigma)

    nll   = compute_nll(X, pi, mus, sigmas)
    delta = abs(nll_history[-1] - nll)
    nll_history.append(nll)

    if iteration % 10 == 0:
        print(f"  iter {iteration:3d} | NLL={nll:.4f} | delta={delta:.2e}")

    if delta < 1e-5 and iteration > 10:
        print(f"Converged at iteration {iteration}!")
        break

gmm_raw_labels = np.argmax(gamma, axis=1)

# Remap by majority-vote against ground truth
remap_gmm = {}
for k in range(3):
    mask = gmm_raw_labels == k
    counts = np.bincount(y_true[mask], minlength=3)
    remap_gmm[k] = np.argmax(counts)
gmm_labels = np.array([remap_gmm[l] for l in gmm_raw_labels])

remapped_snaps = []
for it, raw in snapshots:
    remapped_snaps.append((it, np.array([remap_gmm[l] for l in raw])))

cm_gmm  = confusion_matrix(y_true, gmm_labels, labels=[0, 1, 2])
acc_gmm = np.trace(cm_gmm) / cm_gmm.sum()
print(f"GMM Accuracy: {acc_gmm:.1%}")


# In[44]:


# GMM scatter
fig, ax = plt.subplots(figsize=(7, 6))
for cls in range(3):
    mask = gmm_labels == cls
    ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[cls], label=CLASS_NAMES[cls], s=30, alpha=0.8)
ax.set_title('GMM/EM Final Assignments')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend()
plt.tight_layout()
plt.savefig('/Users/tamarajafar/Desktop/EE541/EE541_HW4/Q1/fig_gmm_scatter.png', dpi=150)
plt.close()


# In[45]:


# GMM confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm_gmm, cmap='Blues')
ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'GMM/EM Confusion Matrix (acc={acc_gmm:.1%})')
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm_gmm[i, j], ha='center', va='center', fontsize=12,
                color='white' if cm_gmm[i, j] > cm_gmm.max() / 2 else 'black')
plt.tight_layout()
plt.savefig('/Users/tamarajafar/Desktop/EE541/EE541_HW4/Q1/fig_gmm_confusion.png', dpi=150)
plt.close()
print("GMM figures saved")


# In[47]:


#ITERATON FIG
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (it, labels) in zip(axes, remapped_snaps):
    for cls in range(3):
        mask = labels == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[cls],
                   label=CLASS_NAMES[cls], s=20, alpha=0.8)
    ax.set_title(f'EM Iteration {it}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    if it == 1:
        ax.legend(fontsize=7)
fig.suptitle('GMM/EM -- First 4 Iterations', fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/tamarajafar/Desktop/EE541/EE541_HW4/Q1/fig_gmm_iterations.png', dpi=150)
plt.close()
print("Iterations figure saved")


# In[50]:


#NLL CONVERGENCE
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(nll_history, color='steelblue', linewidth=2)
ax.set_xlabel('EM Iteration')
ax.set_ylabel('Negative Log-Likelihood')
ax.set_title('EM Convergence')
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('/Users/tamarajafar/Desktop/EE541/EE541_HW4/Q1/fig_nll.png', dpi=150)
plt.close()
print("NLL figure saved")

print(f"\n=== FINAL RESULTS ===")
print(f"  K-Means Accuracy: {acc_km:.1%}")
print(f"  GMM/EM Accuracy:  {acc_gmm:.1%}")


# In[ ]:


#part a vs. b comparison
'''
K-Means (83.9% accuracy) assigns each point to its nearest centroid using straight-line Voronoi boundaries, 
implicitly assuming all 3 clusters are roughly circular and equal in size. 
In the scatter plot, this produces visibly incorrect assignments along the boundary between the Head cluster 
and both ear.  
The confusion matrix confirms that most miscatergarization errors occur at these boundary regions.
GMM/EM (99.6% accuracy) models each cluster as a full multivariate Gaussian with its own covariance matrix,
allowing it to learn that Head is a large, diffuse distribution while the two ears are tight, compact ones. 
The resulting scatter plot shows clean, well-separated regions with only 2 out of 490 points wrong.
The most obvious visual difference between the two plots is along the Head/Ear boundary.
In the K-Means plot, the transition between blue (Head) and red/green (ears) is jagged and irregular, 
with clear misassignments. In the GMM plot, the same boundary is smooth and geometrically consistent 
with the true cluster shapes.
GMM/EM performs significantly better on this dataset because the clusters violate K-Means
core assumption of equal-sized spherical clusters. 
Head contains 290 points spread over a large area while each ear contains only 100 tightly packed points. 
K-Means has no way to represent this difference in density and spread, while GMM learns it directly 
through the covariance matrices. This makes GMM the clearly superior algorithm for this particular dataset.
'''

