# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
# python3 repair_retrain_exp.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --pretrained ./runs/DeepInspect_1/model_best.pth.tar --checkmodel
# python3 repair_retrain_exp.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --expname ResNet50 --epochs 60 --beta 1.0 --cutmix_prob 1.0 --pretrained ./runs/DeepInspect_1/model_best.pth.tar --first 3 --second 5

# python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet_2_4/model_best.pth.tar --lam 0 --extra 256
# set extra batch size same as batch size for half half assumption in new batchnorm layer
import argparse
import os
import shutil
import time
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../common"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../common/CutMix-PyTorch"))
import resnet as RN
import pyramidnet as PYRM
import utils
import numpy as np
import random
import warnings
from tqdm import tqdm
from newbatchnorm2 import dnnrepair_BatchNorm2d
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

torch.manual_seed(124)
torch.cuda.manual_seed(124)
np.random.seed(124)
random.seed(124)
# torch.backends.cudnn.enabled=False
# torch.backends.cudnn.deterministic=True

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(
    description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
parser.add_argument(
    '--original', default='/set/your/model/path', type=str, metavar='PATH')
parser.add_argument(
    '--fix', default='/set/your/model/path', type=str, metavar='PATH')
parser.add_argument('--expid', default="0", type=str, help='experiment id')
parser.add_argument('--checkmodel', help='Check model accuracy',
                    action='store_true')
parser.add_argument('--lam', default=1, type=float,
                    help='hyperparameter lambda')
parser.add_argument('--first', default=3, type=int,
                    help='first object index')
parser.add_argument('--second', default=5, type=int,
                    help='second object index')
parser.add_argument('--extra', default=10, type=int,
                    help='extra batch size')
parser.add_argument('--keeplr', help='set lr 0.001 ',
                    action='store_true')

parser.add_argument('--replace', help='replace bn layer ',
                    action='store_true')

parser.add_argument('--ratio', default=0.5, type=float,
                    help='target ratio for batchnorm layers')
# parser.add_argument('--forward', default=1, type=int,
#                    help='extra batch size')
parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=False)

best_loss = 100
best_err1 = 100
best_err5 = 100
global_epoch_confusion = []


def log_print(var):
    print("logging filter: " + str(var))


glob_bn_total = 0
glob_bn_count = 0


def count_bn_layer(module):
    global glob_bn_total
    for child_name, child in module.named_children():
        if isinstance(child, torch.nn.modules.batchnorm.BatchNorm2d):
            #setattr(module, child_name, nn.Softplus())
            glob_bn_total += 1
        else:
            count_bn_layer(child)


def replace_bn(module):
    global glob_bn_count
    global glob_bn_total
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present

    for child_name, child in module.named_children():
        if isinstance(child, torch.nn.modules.batchnorm.BatchNorm2d):
            glob_bn_count += 1
            if glob_bn_count >= glob_bn_total - 2:  # unfreeze last 3
                print('replaced: bn')
                #new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, 0.5, child.eps, child.momentum, child.affine, track_running_stats=True)
                #new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, 9/19, child.eps, 0.19, child.affine, track_running_stats=True)
                #new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, 9/19, child.eps, child.momentum, child.affine, track_running_stats=True)
                new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, args.ratio, child.eps, child.momentum, child.affine, track_running_stats=True)
                setattr(module, child_name, new_bn)
            else:
                print('replaced: bn')
                new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, 1, child.eps, child.momentum, child.affine, track_running_stats=True)
                setattr(module, child_name, new_bn)
        else:
            replace_bn(child)

def set_bn_eval(model):
    global glob_bn_count
    global glob_bn_total
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            glob_bn_count += 1
            if glob_bn_count < glob_bn_total - 2:  # unfreeze last 3
                # if glob_bn_count < glob_bn_total:# unfreeze last bn
                # if glob_bn_count != glob_bn_total//2:# unfreeze middle bn
                # if glob_bn_count != 1: # unfreeze first bn layer
                # if glob_bn_count < glob_bn_total*2/3:# unfreeze last 1/3
                # if glob_bn_count > glob_bn_total*1/3:# unfreeze first 1/3
                # if glob_bn_count > glob_bn_total*2/3 or glob_bn_count < glob_bn_total*1/3: # unfreeze middle 1/3
                module.eval()
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
            else:
                module.momentum = 0.5


def set_bn_train(model):  # unfreeze all bn
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(True)
            #print("set bn")
            module.train()


def get_dataset_from_specific_classes(target_dataset, first, second):
    first_indices = np.where(np.array(target_dataset.targets) == first)[0]
    second_indices = np.where(np.array(target_dataset.targets) == second)[0]
    target_idx = np.hstack([first_indices, second_indices])
    target_dataset.targets = np.array(target_dataset.targets)[target_idx]
    target_dataset.data = target_dataset.data[target_idx]
    return target_dataset


def main():
    global args, best_err1, best_err5, global_epoch_confusion, best_loss
    args = parser.parse_args()
    assert os.path.isfile(args.pretrained)

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        transform_test2 = transforms.Compose([
            transforms.ToTensor(),
        ])

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=True,
                                  download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=False,
                                  transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=True,
                                 download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False,
                                 transform=transform_test),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False,
                                 transform=transform_test2),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            numberofclass = 10

        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model1 = RN.ResNet(args.dataset, args.depth,
                          numberofclass, args.bottleneck)  # for ResNet
        model2 = RN.ResNet(args.dataset, args.depth,
                          numberofclass, args.bottleneck)  # for ResNet
    
    model1 = torch.nn.DataParallel(model1).cuda()

    model2 = torch.nn.DataParallel(model2).cuda()
    if os.path.isfile(args.original):
        print("=> loading checkpoint '{}'".format(args.original))
        checkpoint = torch.load(args.original)
        model1.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.original))

    if os.path.isfile(args.fix):
        print("=> loading checkpoint '{}'".format(args.fix))
        checkpoint = torch.load(args.fix)
        model2.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.fix))
    '''
    # replace bn layer
    if args.replace:
        model.to('cpu')
        global glob_bn_count
        global glob_bn_total
        glob_bn_total = 0
        glob_bn_count = 0
        count_bn_layer(model)
        print("total bn layer: " + str(glob_bn_total))
        glob_bn_count = 0
        replace_bn(model)
        print(model)
        model = model.cuda()
    '''
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    optimizer = torch.optim.SGD(model2.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True
    #validate(val_loader, model, criterion, 0)

    # for checking pre-trained model accuracy and confusion
    if args.checkmodel:
        global_epoch_confusion.append({})
        get_images(val_loader, test_loader, model1, model2, criterion)

        exit()



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



def get_images(val_loader, test_loader, model1, model2, criterion, epoch=-1):

    from itertools import cycle
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model1.eval()
    correct = 0
    total = 0
    yhats = []
    labels = []
    images = []
    for i, (input, target) in enumerate(test_loader):
        for i in range(len(input)):
            images.append(input[i].cpu().data.numpy())
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model1(input)
        _, top1_output = output.max(1)
        total += target.size(0)
        correct += top1_output.eq(target).sum().item()

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.mean().item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))



        for i in range(len(input)):
            yhats.append(int(top1_output[i].cpu().data.numpy()))
            labels.append(int(target[i].cpu().data.numpy()))
            #images.append(input2[i].cpu().data.numpy())

    model1.eval()
    correct = 0
    total = 0
    yhats2 = []
    labels2 = []

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model2(input)
        _, top1_output = output.max(1)
        total += target.size(0)
        correct += top1_output.eq(target).sum().item()

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.mean().item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))



        for i in range(len(input)):
            yhats2.append(int(top1_output[i].cpu().data.numpy()))
            labels2.append(int(target[i].cpu().data.numpy()))
    
    assert labels == labels2

    for y1,l1,y2,l2,img in zip(yhats,labels,yhats2,labels,images):
        if l1 == args.first and y1==args.second and y2 == args.first:
            #img = img / 2 + 0.5
            img = np.transpose(img, (1,2,0))
            #img = torchvision.utils.make_grid(img)
            plt.imshow(img)
            plt.show()
    return


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        print("saving best model...")
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) +
                        'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global global_epoch_confusion
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * \
            (0.1 ** (epoch // (args.epochs * 0.75)))
        if args.keeplr:
            lr = 0.001
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    global_epoch_confusion[-1]["lr"] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
