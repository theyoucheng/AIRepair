#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import torch


# In[2]:


import pickle


# In[4]:

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def distance(data1, data2):
    return np.linalg.norm(data1 - data2)

def get_confusion(predict, labels):
    confusion = {}
    for l1 in range(10):
        confusion[(l1, l1)] = 0
        for l2 in range(10):
            if l1 == l2 or ((l1, l2) in confusion):
                continue
            c = 0
            subcount = 0
            for i in range(len(predict)):

                if l1 == labels[i] and l2 == predict[i]:
                    c = c + 1

                if l1 == labels[i]:
                    subcount = subcount + 1

            confusion[(l1, l2)] = c/subcount
    return confusion

original_labels = np.load("original_labels.npy")
original_outputs = np.load("original_outputs.npy")
fixed_labels = np.load("fixed_labels.npy")
fixed_outputs = np.load("fixed_outputs.npy")

original_train_labels = np.load("original_train_labels.npy")
original_train_outputs = np.load("original_train_outputs.npy")
fixed_train_labels = np.load("fixed_train_labels.npy")
fixed_train_outputs = np.load("fixed_train_outputs.npy")


original_predict = np.argmax(original_outputs, axis=1)
fixed_predict = np.argmax(fixed_outputs, axis=1)
original_acc = sum(original_predict == original_labels)/10000
fixed_acc = sum(fixed_predict == fixed_labels)/10000
original_train_predict = np.argmax(original_train_outputs, axis=1)
fixed_train_predict = np.argmax(fixed_train_outputs, axis=1)
print("original accuracy: " + str(original_acc))
print("fixed accuracy: " + str(fixed_acc))


original = [[] for _ in range(10)]
original_bar = [0 for _ in range(10)]
#print(list(original_labels))

for (output, label) in zip(original_outputs, original_labels):
    original[label].append(output)
#print(original[5][:5])
original_nn = []
'''
for d in tqdm(original[5]):
    j = 0
    temp_dist = -1
    for i in range(len(original_train_outputs)):
        dist = distance(d, original_train_outputs[i])
        if (dist > 0) and (dist < temp_dist or temp_dist == -1):
            temp_dist = dist
            j = i
    original_bar[original_train_predict[j]] += 1

'''
original_vectors = []
for i in range(10):
    original_vectors.append(np.mean(original[i], axis=0))

# nearest neighbors
'''
ratios = ["0.2", "0.4", "0.6", "0.8", "1"]
fixed_bar = []
for r in ratios:
    fixed_bar.append([0 for _ in range(10)])
    fixed_train_labels = np.load("fixed_" + r + "_train_labels.npy")
    fixed_train_outputs = np.load("fixed_" + r + "_train_outputs.npy")
    fixed_labels = np.load("fixed_" + r + "_labels.npy")
    fixed_outputs = np.load("fixed_" + r + "_outputs.npy")
    fixed_train_predict = np.argmax(fixed_train_outputs, axis=1)

    fixed = [[] for _ in range(10)]
    for (output, label) in zip(fixed_outputs, fixed_labels):
        fixed[label].append(output)

    fixed_vectors = []
    for i in range(10):
        fixed_vectors.append(np.mean(fixed[i], axis=0))
    #fixed_vectors = np.mean(fixed, axis=1)


    fixed_nn = []
    temp_dist = -1
    for d in tqdm(fixed[5]):
        j = 0
        temp_dist = -1
        for i in range(len(fixed_train_outputs)):
            dist = distance(d, fixed_train_outputs[i])
            if (dist > 0) and (dist < temp_dist or temp_dist == -1):
                temp_dist = dist
                j = i
        fixed_bar[-1][fixed_train_predict[j]] += 1

    print(fixed_bar[-1])
    X = np.arange(10)

plt.bar(X + 0.00, original_bar, width = 0.1)
plt.bar(X + 0.10, fixed_bar[0], width = 0.1)
plt.bar(X + 0.20, fixed_bar[1], width = 0.1)
plt.bar(X + 0.30, fixed_bar[2], width = 0.1)
plt.bar(X + 0.40, fixed_bar[3], width = 0.1)
plt.bar(X + 0.50, fixed_bar[4], width = 0.1)
plt.legend(labels=['Original', 'fixed_0.2', 'fixed_0.4', 'fixed_0.6', 'fixed_0.8', 'fixed_1'])
plt.title("Distribution of dogs(5)'s closest neighbors")
plt.xlabel("labels")
plt.ylabel("number of data points")
plt.xticks(X, X)
plt.show()
'''

# confusion
confusion_bar = []
original_confusion = get_confusion(original_predict, original_labels)
confusion_bar.append([0 for _ in range(10)])
for i in range(10):
    confusion_bar[-1][i] = original_confusion[(5, i)]

ratios = ["0.2", "0.4", "0.6", "0.8", "1"]
for r in ratios:
    fixed_train_labels = np.load("fixed_" + r + "_train_labels.npy")
    fixed_train_outputs = np.load("fixed_" + r + "_train_outputs.npy")
    fixed_labels = np.load("fixed_" + r + "_labels.npy")
    fixed_outputs = np.load("fixed_" + r + "_outputs.npy")
    fixed_train_predict = np.argmax(fixed_train_outputs, axis=1)
    fixed_predict = np.argmax(fixed_outputs, axis=1)
    fixed_confusion = get_confusion(fixed_predict, fixed_labels)
    confusion_bar.append([0 for _ in range(10)])
    for i in range(10):
        confusion_bar[-1][i] = fixed_confusion[(5, i)]

    print(confusion_bar[-1])
    X = np.arange(10)

plt.bar(X + 0.00, confusion_bar[0], width = 0.1)
plt.bar(X + 0.10, confusion_bar[1], width = 0.1)
plt.bar(X + 0.20, confusion_bar[2], width = 0.1)
plt.bar(X + 0.30, confusion_bar[3], width = 0.1)
plt.bar(X + 0.40, confusion_bar[4], width = 0.1)
plt.bar(X + 0.50, confusion_bar[5], width = 0.1)
plt.legend(labels=['Original', 'fixed_0.2', 'fixed_0.4', 'fixed_0.6', 'fixed_0.8', 'fixed_1'])
plt.title("Misclassification dogs(5) -> others")
plt.xlabel("labels")
plt.ylabel("Confusion")
plt.xticks(X, X)
plt.show()