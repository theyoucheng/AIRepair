python3 repair_confusion_exp_newbn_landscape_target.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 30 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 0 --gradname "grads_01.pkl" --keeplr 0.001

python3 repair_confusion_exp_newbn_landscape_target.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 30 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 0 --gradname "grads_02.pkl" --keeplr 0.002

python3 repair_confusion_exp_newbn_landscape_target.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 30 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 0 --gradname "grads_005.pkl" --keeplr 0.0005

python3 repair_confusion_exp_newbn_landscape_target.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 30 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 0 --gradname "grads_001.pkl" --keeplr 0.0001



python3 repair_confusion_exp_newbn_landscape_target.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 30 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 1 --gradname "grads_batch_01.pkl" --keeplr 0.001

python3 repair_confusion_exp_newbn_landscape_target.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 30 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 1 --gradname "grads_batch_02.pkl" --keeplr 0.002

python3 repair_confusion_exp_newbn_landscape_target.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 30 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 1 --gradname "grads_batch_005.pkl" --keeplr 0.0005

python3 repair_confusion_exp_newbn_landscape_target.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 30 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --lam 0 --extra 128 --replace --ratio 1 --gradname "grads_batch_001.pkl" --keeplr 0.0001