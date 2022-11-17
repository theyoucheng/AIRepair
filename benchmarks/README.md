# Benchmarks

This folder is used to store different neural network repair tools and frameworks for testing and evaluating.

## Overview

The state-of-the-art open-sourced neural network repair tools are listed below. You can check their GitHub repo to download them. 

| Method                             | Name                     | Experiment goals                                 | Environment                          | Supported Datasets                                                                          |
| :--------------------------------- | :----------------------- | :----------------------------------------------- | :----------------------------------- | :------------------------------------------------------------------------------------------ |
| Retraining and <br>  Refinement    | apricot                  | Accuracy                                         | Keras, tensorflow                    | CIFAR-10                                                                                    |
|                                    | ART                      | Accuracy, Safety                                 | pytorch                              | ACAS\_Xu, Collision Detection                                                               |
|                                    | DeepRepair               | Accuracy                                         | pytorch                              | CIFAR-10, CIFAR-100,                                                                        |
|                                    | DeepFault                | Accuracy, Loss                                   | Keras 2\.3.1 <br>  Tensorflow 1.13.2 | MNIST, CIFAR-10                                                                             |
|                                    | MODE                     | Accuracy                                         | tensorflow1                          | MNIST, <br>  CIFAR-10                                                                       |
|                                    | Ruler                    | Accuracy, Fairness                               | Pytorch                              | Census Income, <br>  German Credit, <br>  Bank Marketing                                    |
|                                    | DNNPatching              | The behaviors in simulation                      | pytorch                              | Driving simulation                                                                          |
| Direct weight <br>  Modification   | Arachne                  | Accuracy, Fairness                               | TensorFlow, pytorch                  | CIFAR-10, GTSRB, F-MNIST, <br>  Twitter US airline sentiment                                |
|                                    | MinimalDNNModification   | Accuracy, Watermark                              | TensorFlow, Marabou                  | ACAS\_Xu, MNIST                                                                             |
|                                    | NNRepair                 | Accuracy                                         | Z3, java                             | MNIST, CIFAR-10                                                                             |
|                                    | NNSynthesizer            | Accuracy                                         | Pytorch, z3                          | XOR-B and blobs network                                                                     |
|                                    | REASSURE                 | Accuracy                                         | Pytorch, gurobi                      | ACAS\_Xu                                                                                    |
|                                    | Socrates                 | Fairness, <br>  Backdoor Safety, <br>  Accuracy  | Pytorch, gurobi                      | Census Income, <br>  German Credit, <br>  Bank Marketing, <br>  GTSRB, <br>  MNIST, F-MNIST |
| Attaching <br>  Repair <br>  Units | AUTOTRAINER              | Accuracy                                         | Tensorflow, keras                    | MNIST,F-MNIST, <br>  CIFAR-10, CIFAR-100                                                    |
|                                    | DeepCorrect              | Accuracy                                         | Python2\.7, <br>  Theano, keras      | CIFAR-100, Imagenet                                                                         |
|                                    | DL2                      | Accuracy, Constr. acc                            | Pytorch                              | CIFAR10 <br>  CIFAR100 <br>  MNIST, F-MNIST                                                 |
|                                    | self-correcting-networks | Accuracy                                         | tensorflow                           | ACAS\_Xu, <br>  Collision detection, <br>  CIFAR-100                                        |
|                                    | PRDNN                    | Efficacy, Drawdown,  <br>  Generalization, Time | Pytorch, gurobi                      | ImageNet, ACAS\_Xu, MNIST                                                                   |
