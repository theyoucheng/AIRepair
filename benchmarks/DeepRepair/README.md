# deeprepair

## Environment
```
conda create -n myenv1 python=3.8
conda install pytorch torchvision tqdm scipy
```

```
conda create -n myenv2 python=2.7
conda install pytorch torchvision tqdm scipy scikit-learn matplotlib
```

## [CIFAR-10](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10/confusion) prototype  


## [COCO](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco/confusion_and_bias) prototype  


## Draw 1 VS rest pairwise confusion (cifar10 vgg)
### Save NPY
#### orig
```
python3 repair_confusion_exp_newbn_vggbn.py --dataset cifar10 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --first 3 --second 5 --checkmodel --expname orig --save_npy
```
#### finetune
```
python3 repair_confusion_exp_newbn_softmax_vggbn.py --dataset cifar10 --pretrained ../../../runs/cifar10_confusion_3_5_2_1/cifar10_vggbn-11_confusion_w-aug_1.0/model_best.pth.tar --first 3 --second 5 --checkmodel --expname finetune --save_npy
```
#### w-aug 0.9
```
python3 repair_confusion_exp_newbn_softmax_vggbn.py --dataset cifar10 --pretrained ../../../runs/cifar10_confusion_3_5_2_0/cifar10_vggbn-11_confusion_w-aug_0.9/model_best.pth.tar --first 3 --second 5 --checkmodel --expname w-aug --save_npy
```
#### w-bn 0.7
```
python3 repair_confusion_exp_newbn_softmax_vggbn.py --dataset cifar10 --pretrained ../../../runs/cifar10_vggbn-11_confusion_w-bn_0.7/model_best.pth.tar --first 3 --second 5 --checkmodel --expname w-bn --save_npy
```
#### w-loss 0.9
```
python3 repair_confusion_exp_newbn_softmax_vggbn.py --dataset cifar10 --pretrained ../../../runs/cifar10_vggbn-11_confusion_w-loss_0.9/model_best.pth.tar --first 3 --second 5 --checkmodel --expname w-loss --save_npy
```
#### w-os 0.001
```
python3 repair_confusion_exp_newbn_softmax_vggbn.py --dataset cifar10 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --first 3 --second 5 --checkmodel --eta 0.001  --expname w-os --save_npy
```
#### w-dbr 0.1
```
python3 repair_confusion_exp_newbn_softmax_vggbn.py --dataset cifar10 --pretrained ../../../runs/cifar10_vggbn-11_confusion_w-dbr_0.1/model_best.pth.tar --first 3 --second 5 --checkmodel --expname w-dbr --save_npy
```

### Draw 1 VS rest pairwise confusion
```
python compute_distribution_cifar.py
```

## Draw Heatmap
### cifar10 vgg
```
python3 repair_confusion_exp_newbn_vggbn.py --dataset cifar10 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --first 3 --second 5 --checkmodel --checkmodel_mode all
```


## Check Model Confusion
### cifar10
```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --first 3 --second 5 --checkmodel
```

### cifar10 vgg
```
python3 repair_confusion_exp_newbn_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --first 3 --second 5 --checkmodel
```

### cifar10 mobilenetv2
```
python3 repair_confusion_exp_newbn.py --net_type mobilenetv2 --dataset cifar10 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --first 3 --second 5 --checkmodel
```

### cifar100
```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar100 --depth 34 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --first 35 --second 98 --checkmodel
```

### coco
```
python2 repair_confusion_bn.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --first "person" --second "bus" --checkmodel
```

### coco gender
```
python2 repair_confusion_bn.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --first "handbag" --second "woman" --checkmodel
```

### cifar10 2 pairs
```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128 --checkmodel --pair1a 3 --pair1b 5 --pair2a 1 --pair2b 9
```

### coco 2 pairs
```
python2 repair_confusion_bn.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --pair1a "person" --pair1b "bus" --pair2a "mouse" --pair2b "keyboard" --checkmodel
```

## Check Model Bias
### cifar10
```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --first 5 --second 3 --third 2 --checkmodel
```

### cifar10 vgg
```
python3 repair_bias_exp_newbn_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --first 5 --second 3 --third 2 --checkmodel
```

### cifar10 mobilenetv2
```
python3 repair_bias_exp_newbn.py --net_type mobilenetv2 --dataset cifar10 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --first 3 --second 5 --third 2 --checkmodel
```

### cifar100
```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar100 --depth 34 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --first 98 --second 35 --third 11 --checkmodel
```

### coco
```
python2 repair_bias_bn.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --first "bus" --second "person" --third "clock" --checkmodel
```

### coco gender
```
python2 repair_bias_bn.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --first "skis" --second "woman" --third "man" --checkmodel
```


## Get Instance
### [cifar10 get instance](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/cifar10_get_instance.py)

```
python3 cifar10_get_instance.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test  --cutmix_prob 0 --original ./runs/cifar10_resnet18_2_4/model_best.pth.tar --fix ./runs/cifar10_resnet_2_4_dogcat_dbr/model_best.pth.tar  --checkmodel --first 5 --second 3

python3 cifar10_get_instance.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test  --cutmix_prob 0 --original ./runs/cifar10_resnet18_2_4/model_best.pth.tar --fix ./runs/cifar10_resnet_2_4_dogcat_dbr/model_best.pth.tar  --checkmodel --first 3 --second 5

```

### [cifar100 get instance](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/cifar10_get_instance.py)

```
python3 cifar10_get_instance.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test  --cutmix_prob 0 --original ./runs/cifar100_resnet34/model_best.pth.tar --fix ./runs/cifar100_resnet34_oversampling/model_best.pth.tar  --checkmodel --first 35 --second 98

python3 cifar10_get_instance.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test  --cutmix_prob 0 --original ./runs/cifar100_resnet34/model_best.pth.tar --fix ./runs/cifar100_resnet34_oversampling/model_best.pth.tar  --checkmodel --first 98 --second 35

```

### [coco get instance](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/get_instance.py)

```
python2 get_instance.py --original original_model/model_best.pth.tar --fix fix_model/model_best.pth.tar --log_dir coco_confusion_repair_aug --first "person" --second "bus" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/'
python2 get_instance.py --original original_model/model_best.pth.tar --fix fix_model/model_best.pth.tar --log_dir coco_confusion_repair_aug --first "bus" --second "person" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/'
```

### [coco gender get instance bias](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/get_instance_coco_gender_bias.py)

```
python2 get_instance_coco_gender_bias.py --original original_model/model_best.pth.tar --fix fix_model/model_best.pth.tar --log_dir coco_confusion_repair_aug --first "skis" --second "man" --third "woman" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/'
```


## Fine-tuning Model
### cifar10
```
python3 repair_confusion_exp_oversampling.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_finetuned --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128 --first 3 --second 5 --weight 1
```
### cifar10 vgg
```
python3 repair_confusion_exp_oversampling_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vggbn_2_4_finetuned --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --extra 128 --first 3 --second 5 --weight 1
```
### cifar10 mobilenetv2
```
python3 repair_confusion_exp_oversampling.py --net_type mobilenetv2 --dataset cifar10 --batch_size 128 --lr 0.1 --expname cifar10_mobilenetv2_finetuned --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --extra 128 --first 3 --second 5 --weight 1
```
### cifar100
```
python3 repair_confusion_exp_oversampling.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_finetuned --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --extra 128 --first 35 --second 98 --weight 1
```
### coco
```
python2 repair_bias_exp_weighted_loss.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --log_dir coco_original_model_finetuned --first "bus" --second "person" --third "clock" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --weight 1 --class_num 80
```
### coco gender
```
python2 repair_confusion_exp_weighted_loss.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --log_dir cocogender_original_model_finetuned --first "handbag" --second "woman" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --weight 1 --class_num 81
```




## [CIFAR-10](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10) experiments
### [CIFAR-10 confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10/confusion) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/train_baseline.py):    

```
python3 train_baseline.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet18_2_4 --epochs 300 --beta 1.0 --cutmix_prob 0
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_oversampling.py):  

```
python3 repair_confusion_exp_oversampling.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128 --first 3 --second 5 --weight 0.3
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_newbn.py):  

```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128 --replace --first 3 --second 5 --ratio 0.2  
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_newbn_softmax.py):
```
python3 repair_confusion_exp_newbn_softmax.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128 --eta 0.3 --checkmodel --first 3 --second 5
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss.py):
```
python3 repair_confusion_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128 --first 3 --second 5 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_dbr.py):  

```
python3 repair_confusion_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --first 3 --second 5 --lam 0.7
```

### [CIFAR-10 bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10/bias) experiments  

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_oversampling.py):  

```
python3 repair_bias_exp_oversampling.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128 --first 3 --second 5 --third 2 --weight 0.3
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_newbn.py):  

```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128 --replace --ratio 0.2 --first 3 --second 5 --third 2
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_newbn_softmax.py):
```
python3 repair_bias_exp_newbn_softmax.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128 --eta 0.3 --checkmodel --first 3 --second 5 --third 2
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_weighted_loss.py):
```
python3 repair_bias_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128 --first 3 --second 5 --third 2 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_dbr.py):  

```
python3 repair_bias_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --first 3 --second 5 --third 2 --lam 0.7
```


### [CIFAR-10 2 pair confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10_multipair/confusion) experiments  


#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/cifar10_multipair/repair_confusion_exp_oversampling.py):  

```
python3 repair_confusion_exp_oversampling.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128  --pair1a 3 --pair1b 5 --pair2a 1 --pair2b 9  --weight 0.3
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/cifar10_multipair/repair_confusion_exp_newbn.py):  

```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128 --replace  --pair1a 3 --pair1b 5 --pair2a 1 --pair2b 9   --ratio 0.3
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/cifar10_multipair/repair_confusion_exp_newbn_softmax.py):
```
python3 repair_confusion_exp_newbn_softmax.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128 --eta 0.3 --checkmodel  --pair1a 3 --pair1b 5 --pair2a 1 --pair2b 9  
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/cifar10_multipair/repair_confusion_exp_weighted_loss.py):
```
python3 repair_confusion_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar --extra 128  --pair1a 3 --pair1b 5 --pair2a 1 --pair2b 9   --target_weight 0.3
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/cifar10_multipair/repair_confusion_dbr.py):  

```
python3 repair_confusion_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_resnet18_2_4/model_best.pth.tar  --pair1a 3 --pair1b 5 --pair2a 1 --pair2b 9 --lam 0.3
```




### [CIFAR-10 VGG-BN confusion](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/train_baseline_vggbn.py) experiments
#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/train_baseline.py):    

```
python3 train_baseline_vggbn.py  --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vggbn_2_4 --epochs 300 --beta 1.0 --cutmix_prob 0
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_oversampling_vggbn.py):  

```
python3 repair_confusion_exp_oversampling_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vggbn_oversampling --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --extra 128 --first 3 --second 5 --weight 0.3
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_newbn_vggbn.py):  

```
python3 repair_confusion_exp_newbn_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vggbn_newbn --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --extra 128 --replace --first 3 --second 5 --ratio 0.4  
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_newbn_softmax_vggbn.py):
```
python3 repair_confusion_exp_newbn_softmax_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vggbn_softmax --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --extra 128 --eta 0.1 --checkmodel --first 3 --second 5
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss_vggbn.py):
```
python3 repair_confusion_exp_weighted_loss_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_vggbn_loss --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --extra 128 --first 3 --second 5 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_dbr_vggbn.py):  

```
python3 repair_confusion_dbr_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_vggbn_dbr --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --first 3 --second 5 --lam 0.7
```

### [CIFAR-10 VGG-BN bias](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/train_baseline_vggbn.py) experiments

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_bias_exp_oversampling_vggbn.py):  

```
python3 repair_bias_exp_oversampling_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_bias_vggbn_oversampling --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --extra 128 --first 3 --second 5 --third 2 --weight 0.3
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_newbn_vggbn.py):  

```
python3 repair_bias_exp_newbn_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_bias_vggbn_newbn --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --extra 128 --replace --first 3 --second 5 --third 2 --ratio 0.4  
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_newbn_softmax_vggbn.py):
```
python3 repair_bias_exp_newbn_softmax_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_bias_vggbn_softmax --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --extra 128 --eta 0.1 --checkmodel --first 3 --second 5 --third 2
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_weighted_loss_vggbn.py):
```
python3 repair_bias_exp_weighted_loss_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_bias_vggbn_loss --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --extra 128 --first 3 --second 5 --third 2 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_dbr_vggbn.py):  

```
python3 repair_bias_dbr_vggbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_bias_vggbn_dbr --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_vggbn_2_4/model_best.pth.tar --first 3 --second 5 --third 2 --lam 0.7
```





### [CIFAR-10 MobileNetV2 confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10/confusion) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/train_baseline.py):    

```
python3 train_baseline.py --net_type mobilenetv2 --dataset cifar10 --batch_size 128 --lr 0.1 --expname cifar10_mobilenetv2 --epochs 300 --beta 1.0 --cutmix_prob 0
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_oversampling.py):  

```
python3 repair_confusion_exp_oversampling.py --net_type mobilenetv2 --dataset cifar10 --batch_size 128 --lr 0.1 --expname cifar10_mobilenetv2_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --extra 128 --first 3 --second 5 --weight 0.3
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_newbn.py):  

```
python3 repair_confusion_exp_newbn.py --net_type mobilenetv2 --dataset cifar10 --batch_size 128 --lr 0.1 --expname cifar10_mobilenetv2_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --extra 128 --replace --first 3 --second 5 --ratio 0.2  
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_newbn_softmax.py):
```
python3 repair_confusion_exp_newbn_softmax.py --net_type mobilenetv2 --dataset cifar10 --batch_size 128 --lr 0.1 --expname cifar10_mobilenetv2_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --extra 128 --eta 0.3 --checkmodel --first 3 --second 5
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss.py):
```
python3 repair_confusion_exp_weighted_loss.py --net_type mobilenetv2 --dataset cifar10 --batch_size 128 --lr 0.1 --expname cifar10_mobilenetv2_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --extra 128 --first 3 --second 5 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_dbr.py):  

```
python3 repair_confusion_dbr.py --net_type mobilenetv2 --dataset cifar10 --batch_size 256 --lr 0.1 --expname cifar10_mobilenetv2_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --first 3 --second 5 --lam 0.7
```

### [CIFAR-10 MobileNetV2 bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar10/bias) experiments  

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_oversampling.py):  

```
python3 repair_bias_exp_oversampling.py --net_type mobilenetv2 --dataset cifar10 --batch_size 128 --lr 0.1 --expname cifar10_mobilenetv2_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --extra 128 --first 3 --second 5 --third 2 --weight 0.3
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_newbn.py):  

```
python3 repair_bias_exp_newbn.py --net_type mobilenetv2 --dataset cifar10 --batch_size 128 --lr 0.1 --expname cifar10_mobilenetv2_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --extra 128 --replace --ratio 0.2 --first 3 --second 5 --third 2
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_newbn_softmax.py):
```
python3 repair_bias_exp_newbn_softmax.py --net_type mobilenetv2 --dataset cifar10 --batch_size 128 --lr 0.1 --expname cifar10_mobilenetv2_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --extra 128 --eta 0.3 --checkmodel --first 3 --second 5 --third 2
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_weighted_loss.py):
```
python3 repair_bias_exp_weighted_loss.py --net_type mobilenetv2 --dataset cifar10 --batch_size 128 --lr 0.1 --expname cifar10_mobilenetv2_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --extra 128 --first 3 --second 5 --third 2 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_dbr.py):  

```
python3 repair_bias_dbr.py --net_type mobilenetv2 --dataset cifar10 --batch_size 256 --lr 0.1 cifar10_mobilenetv2_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar10_mobilenetv2/model_best.pth.tar --first 3 --second 5 --third 2 --lam 0.7
```







## [CIFAR-100 confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar100) experiments
### [CIFAR-100 confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar100/confusion) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/train_baseline_cifar100.py):    

```
python3 train_baseline_cifar100.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34 --epochs 300 --beta 1.0 --cutmix_prob 0
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_exp_oversampling.py):  

```
python3 repair_confusion_exp_oversampling.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --extra 128 --first 35 --second 98 --weight 0.3
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_exp_newbn.py):  

```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --extra 128 --replace --first 35 --second 98 --ratio 0.2  
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_exp_newbn_softmax.py):
```
python3 repair_confusion_exp_newbn_softmax.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --extra 128 --eta 0.8 --checkmodel --first 35 --second 98
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_exp_weighted_loss.py):
```
python3 repair_confusion_exp_weighted_loss.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --extra 128 --first 35 --second 98 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/confusion/repair_confusion_dbr.py):  

```
python3 repair_confusion_dbr.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 256 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --first 35 --second 98 --lam 0.1
```

### [CIFAR-100 bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_7/cifar100/bias) experiments  

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_oversampling.py):  

```
python3 repair_bias_exp_oversampling.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --extra 128 --first 98 --second 35 --third 11 --weight 0.3
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_newbn.py):  

```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --extra 128 --replace --ratio 0.4 --first 98 --second 35 --third 11
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_newbn_softmax.py):
```
python3 repair_bias_exp_newbn_softmax.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --extra 128 --eta 0.3 --checkmodel --first 98 --second 35 --third 11
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_exp_weighted_loss.py):
```
python3 repair_bias_exp_weighted_loss.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 128 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --extra 128 --first 98 --second 35 --third 11 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar100/bias/repair_bias_dbr.py):  

```
python3 repair_bias_dbr.py --net_type resnet --dataset cifar100 --depth 34 --batch_size 256 --lr 0.1 --expname cifar100_resnet34_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ../../../models/cifar100_resnet34/model_best.pth.tar --first 98 --second 35 --third 11 --lam 0.1
```










## [COCO](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco) experiments
### [COCO confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/train_epoch_graph.py):    

```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/'
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_confusion_exp_weighted_loss.py):  

```
python2 repair_confusion_exp_weighted_loss.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --log_dir coco_confusion_repair_aug --first "person" --second "bus" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --weight 0.3 --class_num 80
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_confusion_bn.py):  

```
python2 repair_confusion_bn.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --log_dir coco_confusion_repair_bn --first "person" --second "bus" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --replace --ratio 0.4
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained ../../../models/coco_original_model/checkpoint.pth.tar --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode confusion --first "bus" --second "person"
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss.py):
```
python2 repair_confusion_exp_weighted_loss.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --log_dir coco_confusion_repair_loss --first "bus" --second "person" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --weight 1 --class_num 80 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_confusion_dbr.py):  

```
python2 repair_confusion_dbr.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --log_dir coco_confusion_repair_dbr --first "person" --second "bus" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --lam 0.7
```

### [COCO bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco) experiments  

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_bias_exp_weighted_loss.py):  

```
python2 repair_bias_exp_weighted_loss.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --log_dir coco_bias_repair_aug --first "bus" --second "person" --third "clock" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --weight 0.3 --class_num 80
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_bias_bn.py):  

```
python2 repair_bias_bn.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --log_dir coco_bias_repair_bn --first "bus" --second "person" --third "clock" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --replace --ratio 0.4
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained ../../../models/coco_original_model/checkpoint.pth.tar --ann_dir '../../../../coco/annotations' --image_dir '../../../../data/coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode bias --first "bus" --second "person" --third "clock"
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_weighted_loss.py):
```
python2 repair_bias_exp_weighted_loss.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --log_dir coco_bias_repair_loss --first "bus" --second "person" --third "clock" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --weight 1 --class_num 80 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco/repair_bias_dbr.py):  

```
python2 repair_bias_dbr.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --log_dir coco_bias_repair_dbr --first "bus" --second "person" --third "clock" --second "bus" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --lam 0.7
```

## [COCO gender](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco_gender) experiments
### [COCO gender confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco_gender) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/train_epoch_graph.py):    

```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/'
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_weighted_loss.py):  

```
python2 repair_confusion_exp_weighted_loss.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair_aug --first "handbag" --second "woman" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --weight 0.3 --class_num 81
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_bn.py):  

```
python2 repair_confusion_bn.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair_bn --first "handbag" --second "woman" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --replace --ratio 0.4
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode confusion --first "handbag" --second "woman"
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss.py):
```
python2 repair_confusion_exp_weighted_loss.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair_loss --first "handbag" --second "woman" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --weight 1 --class_num 81 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_dbr.py):  

```
python2 repair_confusion_dbr.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --log_dir coco_gender_confusion_repair_dbr --first "handbag" --second "woman" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --lam 0.7
```

### [COCO gender bias](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco_gender) experiments  

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_bias_exp_weighted_loss.py):  

```
python2 repair_bias_exp_weighted_loss.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --log_dir coco_gender_bias_repair_aug --first "skis" --second "woman" --third "man" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --weight 0.3 --class_num 81
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_bias_bn.py):  

```
python2 repair_bias_bn.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --log_dir coco_gender_bias_repair_bn --first "skis" --second "woman" --third "man" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --replace --ratio 0.4
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode bias --first "skis" --second "woman" --third "man"
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/bias/repair_bias_exp_weighted_loss.py):
```
python2 repair_bias_exp_weighted_loss.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --log_dir coco_gender_bias_repair_loss --first "skis" --second "woman" --third "man" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --weight 1 --class_num 81 --target_weight 0.4
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_gender/repair_bias_dbr.py):  

```
python2 repair_bias_dbr.py --pretrained ../../../models/cocogender_original_model/model_best.pth.tar --log_dir coco_bias_repair_dbr --first "skis" --second "woman" --third "man" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --lam 0.7
```  

### [COCO 2 pair confusion](https://github.com/yuchi1989/deeprepair/tree/master/exp_9/coco_multipair) experiments  

#### [orig](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_multipair/get_original_confusion.py):    

```
python2 get_original_confusion.py --log_dir original_model --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --pair1a person --pair1b bus --pair2a mouse --pair2b keyboard --num_epochs 15
```

#### [w-aug](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_multipair/repair_confusion_exp_weighted_loss.py):  

```
python2 repair_confusion_exp_weighted_loss.py --pretrained ../../coco_original_model/model_best.pth.tar --log_dir coco_2pair_confusion_repair_aug  --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --weight 0.3 --class_num 80 --pair1a person --pair1b bus --pair2a mouse --pair2b keyboard --num_epochs 15
```

#### [w-bn](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_multipair/repair_confusion_bn.py):  

```
python2 repair_confusion_bn.py --pretrained ../../coco_original_model/model_best.pth.tar --log_dir coco_2pair_confusion_repair_bn  --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --replace --pair1a person --pair1b bus --pair2a mouse --pair2b keyboard --ratio 0.4 --num_epochs 15
```

#### [w-os](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_multipair/repair_confusion_exp_newbn_softmax.py):
```
python2 coco_feature_space.py --pretrained ../../../models/coco_original_model/checkpoint.pth.tar --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode multipairconfusion --first person --second bus --third mouse --fourth keyboard
```

#### [w-loss](https://github.com/yuchi1989/deeprepair/blob/master/exp_7/cifar10/confusion/repair_confusion_exp_weighted_loss.py):
```
python2 repair_confusion_exp_weighted_loss.py --pretrained ../../coco_original_model/model_best.pth.tar --log_dir coco_2pair_confusion_repair_loss --pair1a person --pair1b bus --pair2a mouse --pair2b keyboard --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --weight 1 --class_num 80 --target_weight 0.4 --num_epochs 15
```
#### [w-dbr](https://github.com/yuchi1989/deeprepair/blob/master/exp_9/coco_multipair/repair_confusion_dbr.py):  

```
python2 repair_confusion_dbr.py --pretrained ../../coco_original_model/model_best.pth.tar --log_dir coco_2pair_confusion_repair_dbr  --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --pair1a person --pair1b bus --pair2a mouse --pair2b keyboard --lam 0.7 --num_epochs 15
```
