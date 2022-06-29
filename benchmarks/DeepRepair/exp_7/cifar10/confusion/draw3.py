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

grads_01 = pickle.load(open("grads_01.pkl", "rb")) # grads.pkl natural loss
grads_02 = pickle.load(open("grads_02.pkl", "rb")) # grads.pkl natural loss
grads_005 = pickle.load(open("grads_005.pkl", "rb")) # grads.pkl natural loss
grads_001 = pickle.load(open("grads_001.pkl", "rb")) # grads.pkl natural loss
grads_batch_01 = pickle.load(open("grads_batch_01.pkl", "rb")) # grads.pkl natural loss
grads_batch_02 = pickle.load(open("grads_batch_02.pkl", "rb")) # grads.pkl natural loss
grads_batch_005 = pickle.load(open("grads_batch_005.pkl", "rb")) # grads.pkl natural loss
grads_batch_001 = pickle.load(open("grads_batch_001.pkl", "rb")) # grads.pkl natural loss


lol_01 = np.array(grads_01).flatten()
lol_02 = np.array(grads_02).flatten()
lol_005 = np.array(grads_005).flatten()
lol_001 = np.array(grads_001).flatten()

kek_01 = []
kek_02 = []
kek_005 = []
kek_001 = []
for i in range(1,len(lol_01)):
  kek_01.append((lol_01[i-1] - lol_01[i]).norm(p=2).item())
  kek_02.append((lol_02[i-1] - lol_02[i]).norm(p=2).item())
  kek_005.append((lol_005[i-1] - lol_005[i]).norm(p=2).item())
  kek_001.append((lol_001[i-1] - lol_001[i]).norm(p=2).item())
  
min_curve = []
max_curve = []
for i in range(len(kek_01)):
    min_curve.append(np.min([kek_01[i], kek_02[i], kek_005[i], kek_001[i]]))
    max_curve.append(np.max([kek_01[i], kek_02[i], kek_005[i], kek_001[i]]))

plt.figure(figsize=(15, 10))
plt.fill_between(range(len(min_curve)//5+1), min_curve[::5], max_curve[::5])
plt.savefig(os.path.join(figures_path, 'block21_baseline.pdf'), dpi=500, quality=100)
plt.figure(figsize=(15, 10))
plt.plot(range(len(kek_01)), kek_01)
plt.savefig(os.path.join(figures_path, 'block22_baseline.pdf'), dpi=500, quality=100)

lol_batch_01 = np.array(grads_batch_01).flatten()
lol_batch_02 = np.array(grads_batch_02).flatten()
lol_batch_005 = np.array(grads_batch_005).flatten()
lol_batch_001 = np.array(grads_batch_001).flatten()

kek_batch_01 = []
kek_batch_02 = []
kek_batch_005 = []
kek_batch_001 = []
for i in range(1,len(lol_01)):
    kek_batch_01.append((lol_batch_01[i-1] - lol_batch_01[i]).norm(p=2).item())
    kek_batch_02.append((lol_batch_02[i-1] - lol_batch_02[i]).norm(p=2).item())
    kek_batch_005.append((lol_batch_005[i-1] - lol_batch_005[i]).norm(p=2).item())
    kek_batch_001.append((lol_batch_001[i-1] - lol_batch_001[i]).norm(p=2).item())
  
min_batch_curve = []
max_batch_curve = []
for i in range(len(kek_01)):
    min_batch_curve.append(np.min([kek_batch_01[i], kek_batch_02[i], kek_batch_005[i], kek_batch_001[i]]))
    max_batch_curve.append(np.max([kek_batch_01[i], kek_batch_02[i], kek_batch_005[i], kek_batch_001[i]]))

plt.figure(figsize=(15, 10))
plt.fill_between(range(len(min_curve)//5+1), min_curve[::5], max_curve[::5])
plt.fill_between(range(len(min_batch_curve)//5+1), min_batch_curve[::5], max_batch_curve[::5])
plt.savefig(os.path.join(figures_path, 'block51_ourbatchnorm.png'), dpi=500, quality=100)
plt.show()




step = 30
steps = np.arange(0, len(min_curve), step)
plt.figure(figsize=(15, 10))
plt.fill_between(steps, min_curve[::step], max_curve[::step],
                alpha=0.5, color='C1', label='confusion + baseline')
plt.plot(steps, min_curve[::step], color='C1')
plt.plot(steps, max_curve[::step], color='C1')

plt.fill_between(steps, min_batch_curve[::step], max_batch_curve[::step],
                alpha=0.5, color='C2', label='confusion + ourbatchNorm')
plt.plot(steps, min_batch_curve[::step], color='C2')
plt.plot(steps, max_batch_curve[::step], color='C2')

plt.legend(fontsize=19)
plt.title('Gradient Predictiveness', fontsize=20)
plt.ylabel('Gradient Predictiveness', fontsize=13)
plt.xlabel('Steps', fontsize=13)
plt.savefig(os.path.join(figures_path, 'gradient_predictiveness.pdf'), dpi=500, quality=100)
plt.show()


step = 50
steps = np.arange(0, len(min_curve), step)
plt.figure(figsize=(15, 10))

plt.plot(steps, max_curve[::step], color='C1', label='confusion + baseline')

plt.plot(steps, max_batch_curve[::step], color='C2', label='confusion + ourbatchNorm')

plt.legend(fontsize=19)
plt.title('Effective beta-smoothness', fontsize=20)
plt.ylabel('Effective beta-smoothness', fontsize=13)
plt.xlabel('Steps', fontsize=13)
plt.savefig(os.path.join(figures_path, 'effective_beta_smoothness.pdf'), dpi=500, quality=100)
plt.show()



max_curve_step = max_curve[::step]
running_std = []
for i in range(1, len(max_curve_step)):
    running_std.append(np.std(max_curve_step[:i]))
plt.plot(steps[1:], running_std, color='C1', label='confusion + baseline')

max_curve_step = max_batch_curve[::step]
running_std2 = []
for i in range(1, len(max_curve_step)):
    running_std2.append(np.std(max_curve_step[:i]))
plt.plot(steps[1:], running_std2, color='C2', label='confusion + ourbatchNorm')
plt.savefig(os.path.join(figures_path, 'ziyuan_required.pdf'), dpi=500, quality=100)
plt.show()