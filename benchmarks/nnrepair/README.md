# NNRepair: Constraint based Repair of Neural Network Classifiers 


## Directories

### Neural Network Models
Neural Network models used in paper are stored under NN-Code
directory. This directory contains model for MNIST-LowQuality,
MNIST-Poisoned, MNIST-Adversarial, Cifar10-Poisoned and Cifar10-Adversarial.


### Code for combining experts 
The code to combine experts is located inside CombinationCode directory.

### Repair constraints generated using SPF 
The constraints file are located in Constraints directory. This directory has 
5 subdirectories i.e., ***MNIST-LowQuality, MNIST-Poisoned, MNIST-Adversarial, 
Cifar10-Poisoned and Cifar10-Adversarial***. Within each models directory, 
we have grouped the constraint files by Intermediate-Layer Repair and Last-Layer Repair. 
The constraints file names follow the naming convention repairFor_label.txt 
(e.g., repairFor1 refers to constraint file for label 1). Please note that we had 4 different
experiments for each repair (0, 10, 50 and 100 passing tests). These 4 scenarios are named as 
ExpA, ExpB, ExpC and ExpD respectively. 
  
### Z3 Solutions for Constraint Files
The constraints file are located in Z3Solutions directory. This directory has 
5 subdirectories i.e., ***MNIST-LowQuality, MNIST-Poisoned, MNIST-Adversarial, 
Cifar10-Poisoned and Cifar10-Adversarial***. Within each models directory, 
we have grouped the solution files by Intermediate-Layer Repair and Last-Layer Repair. 
The solution file names follow the naming convention solution_label.txt 
(e.g., solution1 refers to solution file for label 1). Please note that we had 4 different
experiments for each repair (0, 10, 50 and 100 passing tests). These 4 scenarios are named as 
ExpA, ExpB, ExpC and ExpD respectively. 


### Final Results
The results file are located in Results directory. This directory has 
5 subdirectories i.e., ***MNIST-LowQuality, MNIST-Poisoned, MNIST-Adversarial, 
Cifar10-Poisoned and Cifar10-Adversarial***. Within each models directory, 
we have grouped the results files by Intermediate-Layer Repair and Last-Layer Repair. 
The results file names follow the naming convention model_repairtype_Exp#_dataset.csv 
(e.g., POISONED_CIFAR_LAST_LAYER_ExpA_POISONED_TEST.csv  refers to results on Poisoned Test Dataset
when last layer repair was performed on CIFAR POISONED MODEL). Please note that we had 4 different
experiments for each repair (0, 10, 50 and 100 passing tests). These 4 scenarios are named as 
ExpA, ExpB, ExpC and ExpD respectively. 



