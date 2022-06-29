#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import torch


# In[2]:


import pickle


# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[6]:
figures_path = "/home/yt2667/"

grads_baseline_confusion = pickle.load(open("grads_baseline_confusion.pkl", "rb")) # grads.pkl confusion loss
grads_ourbatchnorm_confusion = pickle.load(open("grads_ourbatchnorm_confusion.pkl", "rb")) # grads.pkl --replace confusion loss


# In[ ]:


lol_batch_0 = np.array(grads_baseline_confusion).flatten()
lol_batch_1 = np.array(grads_ourbatchnorm_confusion).flatten()


# In[ ]:


kek_batch_0 = []
for i in range(1,len(lol_batch_0)):
    kek_batch_0.append((lol_batch_0[i-1] - lol_batch_0[i]).norm(p=2).item())

kek_batch_1 = []
for i in range(1,len(lol_batch_1)):
    kek_batch_1.append((lol_batch_1[i-1] - lol_batch_1[i]).norm(p=2).item())

# In[ ]:


min_batch_curve = []
max_batch_curve = []
for i in range(len(kek_batch_0)):
    min_batch_curve.append(np.min([kek_batch_0[i]]))
    max_batch_curve.append(np.max([kek_batch_0[i]]))


min_batch_curve_1 = []
max_batch_curve_1 = []
for i in range(len(kek_batch_1)):
    min_batch_curve_1.append(np.min([kek_batch_1[i]]))
    max_batch_curve_1.append(np.max([kek_batch_1[i]]))

# In[ ]:


plt.figure(figsize=(15, 10))
plt.fill_between(range(len(min_batch_curve)//5+1), min_batch_curve[::5], max_batch_curve[::5], label='confusion loss + baseline')
plt.fill_between(range(len(min_batch_curve_1)//5+1), min_batch_curve_1[::5], max_batch_curve_1[::5], label='confusion loss + our BatchNorm')
plt.savefig(os.path.join(figures_path, 'loss_landscape.pdf'), dpi=500, quality=100)
#plt.show()


# In[ ]:


step = 30
steps = np.arange(0, len(min_batch_curve), step)
plt.figure(figsize=(15, 10))

plt.fill_between(steps, min_batch_curve[::step], max_batch_curve[::step],
                alpha=0.5, color='C1', label='confusion loss + baseline')
plt.plot(steps, min_batch_curve[::step], color='C1')
plt.plot(steps, max_batch_curve[::step], color='C1')


plt.fill_between(steps, min_batch_curve_1[::step], max_batch_curve_1[::step],
                alpha=0.5, color='C2', label='confusion loss + our BatchNorm')
plt.plot(steps, min_batch_curve_1[::step], color='C2')
plt.plot(steps, max_batch_curve_1[::step], color='C2')

plt.legend(fontsize=19)
plt.title('Gradient Predictiveness', fontsize=20)
plt.ylabel('Gradient Predictiveness', fontsize=13)
plt.xlabel('Steps', fontsize=13)
plt.savefig(os.path.join(figures_path, 'gradient_predictiveness.pdf'), dpi=500, quality=100)
#plt.show()


# In[ ]:


step = 50
steps = np.arange(0, len(max_batch_curve), step)
plt.figure(figsize=(15, 10))

plt.plot(steps, max_batch_curve[::step], color='C1', label='confusion loss + baseline')
plt.plot(steps, max_batch_curve_1[::step], color='C2', label='confusion loss + our BatchNorm')

plt.legend(fontsize=19)
plt.title('Effective beta-smoothness', fontsize=20)
plt.ylabel('Effective beta-smoothness', fontsize=13)
plt.xlabel('Steps', fontsize=13)
plt.savefig(os.path.join(figures_path, 'effective_beta_smoothness.pdf'), dpi=500, quality=100)
#plt.show()
