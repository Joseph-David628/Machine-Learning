#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


from numpy.linalg import inv
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


A = np.matrix('0 2 4; 2 4 2; 3 3 1'). #define matrices
b = np.matrix('-2; -2; -4')
c = np.matrix('1; 1; 1')


# In[4]:


A


# In[5]:


Ainv = inv(A)	#compute inverse
Ainv


# In[6]:


Ainv*b	#compute A^(-1)*b


# In[7]:


A*c	#compute A*c


# In[12]:

#results for A.12(a),(b)
K = [1, 8, 64, 512]
for k in K:
    Y = np.sum(np.sign(np.random.randn(40000,k))*np.sqrt(1./k), axis=1)
    plt.step(sorted(Y), np.arange(1,40001)/float(40000), label = k)
Z = np.random.randn(40000)
plt.step(sorted(Z), np.arange(1,40001)/float(40000), label = 'Gaussian')
plt.xlabel('Observations')
plt.ylabel('Probability')
plt.xlim([-3,3])
plt.legend()





