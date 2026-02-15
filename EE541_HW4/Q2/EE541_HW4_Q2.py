#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np

DEBUG = False   # set True first to test, then False for submission
DATA_FNAME = 'hw4_binary_sequences.hdf5'

if DEBUG:
    num_sequences = 3
    sequence_length = 4
else:
    num_sequences = 10
    sequence_length = 50


# In[2]:


if DEBUG:
    x_list = [
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
    ]
else:
    x_list = [
        # Sequence 1
        [1,0,1,0,1,1,0,1,0,0,0,1,0,1,1,0,0,1,0,0,
         1,1,0,1,0,1,1,0,1,0,1,0,1,1,0,0,1,0,1,1,
         0,1,0,1,1,1,0,1,0,1],
        # Sequence 2
        [0,0,1,1,1,0,0,1,1,0,1,0,1,0,0,1,0,1,1,0,
         0,1,0,0,1,0,1,1,0,1,1,0,1,0,1,0,1,1,0,0,
         1,0,1,0,1,1,0,1,0,1],
        # Sequence 3
        [1,1,0,1,1,0,0,1,0,1,1,1,1,0,0,1,1,0,1,0,
         0,1,0,1,1,0,0,1,1,0,1,0,1,0,1,1,0,0,0,0,
         1,1,0,0,1,0,1,0,1,1],
        # Sequence 4
        [0,1,0,1,0,0,1,0,1,1,0,1,0,1,1,0,1,0,0,1,
         0,1,1,0,0,1,0,1,0,1,1,0,1,1,0,1,0,0,1,0,
         1,0,1,1,0,0,1,0,1,0],
        # Sequence 5
        [1,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,1,1,0,1,
         0,0,1,0,1,1,0,0,0,1,0,1,0,1,0,0,1,0,1,1,
         1,0,1,0,0,1,0,1,1,0],
        # Sequence 6
        [1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,0,1,0,0,1,
         1,0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,1,0,
         0,1,0,0,1,1,0,1,0,0],
        # Sequence 7
        [0,1,0,1,0,1,0,0,1,0,1,0,1,1,0,1,0,1,0,1,
         1,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,1,0,1,
         0,1,0,0,1,0,1,0,1,1],
        # Sequence 8
        [1,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,
         1,1,0,0,1,0,1,0,1,0,0,1,1,0,0,1,0,1,0,1,
         1,0,1,0,0,1,1,0,0,1],
        # Sequence 9
        [1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,1,0,
         1,0,1,1,0,0,1,0,1,1,0,1,1,0,1,0,1,0,1,1,
         0,1,0,1,1,0,1,0,1,0],
        # Sequence 10
        [0,1,1,0,0,1,0,0,1,1,0,1,0,1,0,0,1,1,0,1,
         0,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,1,0,1,0,
         1,0,1,0,1,0,1,0,1,0],
    ]


# In[3]:


human_binary = np.asarray(x_list)

assert human_binary.shape[0] == num_sequences, \
    f'Error: expected {num_sequences} sequences, got {human_binary.shape[0]}'
assert human_binary.shape[1] == sequence_length, \
    f'Error: expected length {sequence_length}, got {human_binary.shape[1]}'

print(f"Shape check passed: {human_binary.shape}")


# In[4]:


with h5py.File(DATA_FNAME, 'w') as hf:
    hf.create_dataset('human_binary', data=human_binary)
print(f"Written to {DATA_FNAME}")


# In[6]:


with h5py.File(DATA_FNAME, 'r') as hf:
    hb_copy = hf['human_binary'][:]

np.testing.assert_array_equal(human_binary, hb_copy)
print("Read-back verification passed!")
print(f"\nFinal array shape: {human_binary.shape}")
print(f"Per-sequence sum of 1s: {human_binary.sum(axis=1).tolist()}")
print(f"Overall fraction of 1s: {human_binary.mean():.3f}  (true random = 0.500)")

