# Benchmarks

This folder is used to store different neural network repair tools and frameworks for testing and evaluating.

## Overview

The state-of-the-art open-sourced neural network repair tools are listed below. You can check their GitHub repo to download them. 

| Method                             | Name                     | Experiment goals                                 | Environment                          | Supported Datasets                                                                          |
| :--------------------------------- | :----------------------- | :----------------------------------------------- | :----------------------------------- | :------------------------------------------------------------------------------------------ |
| Retraining and <br>  Refinement    | [apricot](https://github.com/coinse/arachne/tree/master/apricot)                  | Accuracy                                         | Keras, tensorflow                    | CIFAR-10                                                                                    |
|                                    | [ART](https://github.com/XuankangLin/ART)                      | Accuracy, Safety                                 | pytorch                              | ACAS\_Xu, Collision Detection                                                               |
|                                    | [DeepRepair](https://github.com/yuchi1989/deeprepair)               | Accuracy                                         | pytorch                              | CIFAR-10, CIFAR-100,                                                                        |
|                                    | [DeepFault](https://github.com/hfeniser/DeepFault)                | Accuracy, Loss                                   | Keras 2\.3.1 <br>  Tensorflow 1.13.2 | MNIST, CIFAR-10                                                                             |
|                                    | [MODE](https://github.com/fabriceyhc/mode_nn_debugging)                     | Accuracy                                         | tensorflow1                          | MNIST, <br>  CIFAR-10                                                                       |
|                                    | [Ruler](https://github.com/wssun/RULER)                    | Accuracy, Fairness                               | Pytorch                              | Census Income, <br>  German Credit, <br>  Bank Marketing                                    |
|                                    | [DNNPatching](https://github.com/ApoorvaNandini/Patching)              | The behaviors in simulation                      | pytorch                              | Driving simulation                                                                          |
| Direct weight <br>  Modification   | [Arachne](https://github.com/coinse/arachne)                  | Accuracy, Fairness                               | TensorFlow, pytorch                  | CIFAR-10, GTSRB, F-MNIST, <br>  Twitter US airline sentiment                                |
|                                    | [MinimalDNNModification](https://github.com/jjgold012/MinimalDNNModificationLpar2020)   | Accuracy, Watermark                              | TensorFlow, Marabou                  | ACAS\_Xu, MNIST                                                                             |
|                                    | [NNRepair](https://github.com/nnrepair/nnrepair)                 | Accuracy                                         | Z3, java                             | MNIST, CIFAR-10                                                                             |
|                                    | [NNSynthesizer](https://github.com/dorcoh/NNSynthesizer)            | Accuracy                                         | Pytorch, z3                          | XOR-B and blobs network                                                                     |
|                                    | [REASSURE](https://github.com/BU-DEPEND-Lab/REASSURE)                 | Accuracy                                         | Pytorch, gurobi                      | ACAS\_Xu                                                                                    |
|                                    | [Socrates](https://github.com/longph1989/Socrates)                 | Fairness, <br>  Backdoor Safety, <br>  Accuracy  | Pytorch, gurobi                      | Census Income, <br>  German Credit, <br>  Bank Marketing, <br>  GTSRB, <br>  MNIST, F-MNIST |
| Attaching <br>  Repair <br>  Units | [AUTOTRAINER](https://github.com/shiningrain/AUTOTRAINER)              | Accuracy                                         | Tensorflow, keras                    | MNIST,F-MNIST, <br>  CIFAR-10, CIFAR-100                                                    |
|                                    | [DeepCorrect](https://github.com/tendev5/DeepCorrect)              | Accuracy                                         | Python2\.7, <br>  Theano, keras      | CIFAR-100, Imagenet                                                                         |
|                                    | [DL2](https://github.com/eth-sri/dl2)                      | Accuracy, Constr. acc                            | Pytorch                              | CIFAR10 <br>  CIFAR100 <br>  MNIST, F-MNIST                                                 |
|                                    | [self-correcting-networks](https://github.com/cmu-transparency/self-correcting-networks) | Accuracy                                         | tensorflow                           | ACAS\_Xu, <br>  Collision detection, <br>  CIFAR-100                                        |
|                                    | [PRDNN](https://github.com/95616ARG/PRDNN)                   | Efficacy, Drawdown,  <br>  Generalization, Time | Pytorch, gurobi                      | ImageNet, ACAS\_Xu, MNIST                                                                   |
## Method selection principles

This graph shows the flowchart of choosing appropriate repair method of different pretrained models.

