#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np


# In[15]:


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# In[16]:


# Input dataset
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# Output dataset
y = np.array([[0],
              [1],
              [1],
              [0]])


# In[17]:


# Seed the random number generator
np.random.seed(1)

# Initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1


# In[18]:


# Training loop
for j in range(60000):
    # Forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    
    # Calculate the error
    l2_error = y - l2
    
    # Backpropagation
    l2_delta = l2_error * nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)
    
    # Update weights (gradient descent)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


# In[22]:


print(l2)


# In[ ]:




