# Models

This folder is used to store the pretrained models that need to be repaired.

## Overview

The list below shows the frequently used models in deep neural network repairing:

| Dataset Name | Models     | Type | Scale                                                           |
| :----------- | :--------- | :--- | :-------------------------------------------------------------- |
| [ACAS Xu](https://github.com/stanleybak/vnncomp2021/tree/main/benchmarks/acasxu)      | 36 Unsafe  | FFNN | 5 inputs, 6 layers, 300 neurons,5 outputs                       |
|              | 9 Safe     | FFNN | 5 inputs, 6 layers, 300 neurons 5 outputs                       |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)     | ResNet18   | CNN  | 32\*32 inputs, 18 layers deep, 10 outputs                       |
|              | ResNet34   | CNN  | 32\*32 inputs, 34 layers deep, 10 outputs                       |
| [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)    | ResNet50   | CNN  | 32\*32 inputs, 50 layers deep, 100 outputs <br>  25M parameters |
| [MNIST](http://yann.lecun.com/exdb/mnist/)        | MNIST-CNN  | CNN  |                                                                 |
|              | MNIST-FFNN | FFNN | 28\*28 input, 6 layers, 784 per layer, 10 output                |
| [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)        | FMNIST-CNN  | CNN  |                                                                 |
|              | FMNIST-FFNN | FFNN | 28\*28 input, 6 layers, 784 per layer, 10 output                |
| [Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))       | FFNN       | FFNN | 6 layers FFNN                                                   |
| [bank](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)         | FFNN       | FFNN | 6 layers FFNN                                                   |
| [census](https://www.kaggle.com/datasets/uciml/adult-census-income)       | FFNN       | FFNN | 6 layers FFNN                                                   |
