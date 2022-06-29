## reproduce results on cifar10
### repair state-of-the-art cutmix model
#### step1: clone cutmix model
```
git clone https://github.com/clovaai/CutMix-PyTorch.git
```

#### step2: install the necessary environment from https://github.com/clovaai/CutMix-PyTorch
#### see https://github.com/clovaai/CutMix-PyTorch

#### step3: copy train_baseline.py from src to cutmix folder and train baseline model
```
python3 train_baseline.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_resnet18_2_4 --epochs 300 --beta 1.0 --cutmix_prob 0
```
#### step4: check model cat-dog confusion (copy cifar10_repair_confusion.py from src)
```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --checkmodel
logging filter: 87.07
logging filter: 0.8707
logging filter: pair accuracy: 0.7945
logging filter: 3 -> 5
logging filter: 0.093
logging filter: 5 -> 3
logging filter: 0.091

```
#### step5: repair model to reduce cat-dog confusion
```

python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 0.2

logging filter: 86.3
logging filter: 0.863
logging filter: pair accuracy: 0.7095
logging filter: 3 -> 5
logging filter: 0.042
logging filter: 5 -> 3
logging filter: 0.102


python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 0.5
logging filter: 80.83
logging filter: 0.8083
logging filter: pair accuracy: 0.489
logging filter: 3 -> 5
logging filter: 0.021
logging filter: 5 -> 3
logging filter: 0.061


python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 0.8

logging filter: 72.97
logging filter: 0.7297
logging filter: pair accuracy: 0.2225
logging filter: 3 -> 5
logging filter: 0.004
logging filter: 5 -> 3
logging filter: 0.023

python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 1

logging filter: 67.38
logging filter: 0.6738
logging filter: pair accuracy: 0.0935
logging filter: 3 -> 5
logging filter: 0.0
logging filter: 5 -> 3
logging filter: 0.005
```
