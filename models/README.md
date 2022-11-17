# Models

This folder is used to store the pretrained models that need to be repaired.

## Overview

The list below shows the frequently used models in deep neural network repairing:

| Dataset Name | Models     | Type | Scale                                                           |
| :----------- | :--------- | :--- | :-------------------------------------------------------------- |
| ACAS Xu      | 36 unsafe  | FFNN | 5 inputs, 6 layers, 300 neurons,5 outputs                       |
|              | 9 safe     | FFNN | 5 inputs, 6 layers, 300 neurons 5 outputs                       |
| Cifar-10     | ResNet18   | CNN  | 32\*32 inputs, 18 layers deep, 10 outputs                       |
|              | ResNet34   | CNN  | 32\*32 inputs, 34 layers deep, 10 outputs                       |
| Cifar-100    | ResNet50   | CNN  | 32\*32 inputs, 50 layers deep, 100 outputs <br>  25M parameters |
| MNIST        | MNIST-CNN  | CNN  |                                                                 |
|              | MNIST-FFNN | FFNN | 28\*28 input, 6 layers, 784 per layer, 10 output                |
| Credit       | FFNN       | FFNN | 6 layers FFNN                                                   |
| bank         | FFNN       | FFNN | 6 layers FFNN                                                   |
| census       | FFNN       | FFNN | 6 layers FFNN                                                   |
