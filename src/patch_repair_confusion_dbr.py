# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
# python3 repair_retrain_exp.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --pretrained ./runs/DeepInspect_1/model_best.pth.tar --checkmodel
# python3 repair_retrain_exp.py --net_type resnet --dataset cifar10 --depth 50 --batch_size 256 --lr 0.1 --expname ResNet50 --epochs 60 --beta 1.0 --cutmix_prob 1.0 --pretrained ./runs/DeepInspect_1/model_best.pth.tar --first 3 --second 5

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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../common"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../common/CutMix-PyTorch"))
import resnet as RN
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import random
import warnings
from tqdm import tqdm
from newbatchnorm2 import dnnrepair_BatchNorm2d
from models import MLP, MnistNet


warnings.filterwarnings("ignore")

torch.manual_seed(124)
torch.cuda.manual_seed(124)
np.random.seed(124)
random.seed(124)
#torch.backends.cudnn.enabled=False
#torch.backends.cudnn.deterministic=True

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(
    description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--type', default='confusion', type=str, help='choose the fixing mode: confusion or bias')
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
    '--pretrained', default='/set/your/model/path', type=str, metavar='PATH')
parser.add_argument('--saved_model', required=True, type=str, help='the path and file name to save your repaired models')
parser.add_argument('--expid', default="0", type=str, help='experiment id')
parser.add_argument('--checkmodel', help='Check model accuracy',
    action='store_true')
parser.add_argument('--lam', default=1, type=float,
                    help='hyperparameter lambda')
parser.add_argument('--eta', default=0.3, type=float,
                    help='hyperparameter eta')
parser.add_argument('--first', default=3, type=int,
                    help='first object index')
parser.add_argument('--second', default=5, type=int,
                    help='second object index')
parser.add_argument('--third', default=2, type=int,
                    help='third object index')
parser.add_argument('--extra', default=10, type=int,
                    help='extra batch size')
parser.add_argument('--keeplr', help='set lr 0.001 ',
    action='store_true')
parser.add_argument('--forward', default=1, type=int,
                    help='extra batch size')
parser.add_argument('--replace', help='replace bn layer ',
                    action='store_true')
parser.add_argument('--additional-param', dest='addition', default="",type=str, help = 'additional param for deeprepair, including specify repairing method')
parser.add_argument('--ratio', default=0.5, type=float,
                    help='target ratio for batchnorm layers')
parser.add_argument('--save_npy', help='save npy for analysis',
                    action='store_true')
parser.add_argument('--weight', default=1, type=float,
                    help='oversampling weight')
parser.add_argument('--target_weight', default=1, type=float,
                    help='weights assigned to the original loss and 1-target_weight is the weight assigned to mistakes on the confusion pair in the loss. It get used when smaller than 1.')
args = parser.parse_args()



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
if "exp_newbn" in args.addition:
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
                print('replaced: bn', glob_bn_count)
                new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, args.ratio, child.eps, child.momentum, child.affine, track_running_stats=True)
                setattr(module, child_name, new_bn)
            else:
                if args.addition.contain("newbn"):
                    print('replaced: bn')
                new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, 1, child.eps, child.momentum, child.affine, track_running_stats=True)
                setattr(module, child_name, new_bn)
        else:
            replace_bn(child)

def set_bn_eval_exp(model):
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


def set_bn_train_exp(model):  # unfreeze all bn
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(True)
            #print("set bn")
            module.train()



def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

def set_bn_train(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.train()

def compute_confusion(confusion_matrix, first, second):
    confusion = 0
    if (first, second) in confusion_matrix:
        confusion += confusion_matrix[(first, second)]

    if (second, first) in confusion_matrix:
        confusion += confusion_matrix[(second, first)]
    return confusion/2

def compute_bias(confusion_matrix, first, second, third):
    return abs(compute_confusion(confusion_matrix, first, second) - compute_confusion(confusion_matrix, first, third))

def get_dataset_from_specific_classes(target_dataset, first , second):
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
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10

            target_train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
            target_train_dataset = get_dataset_from_specific_classes(target_train_dataset, args.first, args.second)
            target_test_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=transform_test)
            target_test_dataset = get_dataset_from_specific_classes(target_test_dataset, args.first, args.second)
            target_train_loader = torch.utils.data.DataLoader(target_train_dataset, batch_size=args.extra, shuffle=True,
                                        num_workers=args.workers, pin_memory=True)
            target_val_loader = torch.utils.data.DataLoader(target_test_dataset, batch_size=args.extra, shuffle=True,
                                        num_workers=args.workers, pin_memory=True)
                                        
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))
    elif args.dataset.startswith('fmnist'):
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
        train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
        target_train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(test_data, batch_size=64, shuffle=True)
        target_val_loader = DataLoader(test_data, batch_size=64, shuffle=True)
        numberofclass = 10
    elif args.dataset.startswith('mnist'):
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
        train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
        target_train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(test_data, batch_size=64, shuffle=True)
        target_val_loader = DataLoader(test_data, batch_size=64, shuffle=True)
        numberofclass = 10
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth,
                          numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'mnist':
        model = MnistNet()
    elif args.net_type == 'fmnist':
        model = MnistNet()
    else:
        raise Exception(
            'unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        #model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.pretrained))

    print(model)
    print('the number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
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

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True
    #validate(val_loader, model, criterion, 0)

    # for checking pre-trained model accuracy and confusion
    if args.checkmodel:
        global_epoch_confusion.append({})
        top1err, _, _ = get_confusion(val_loader, model, criterion)
        # cat->dog confusion
        log_print(str(args.first) + " -> " + str(args.second))
        log_print(global_epoch_confusion[-1]["confusion"][(args.first, args.second)])
        # dog->cat confusion
        log_print(str(args.second) + " -> " + str(args.first))
        log_print(global_epoch_confusion[-1]["confusion"][(args.second, args.first)])
        if args.addition.contains("exp_newbn"):
            obj1_count = global_epoch_confusion[-1]["obj1_count"]
            obj2_count = global_epoch_confusion[-1]["obj2_count"]

            first_i = 0
            second_i = 0
            for i in range(numberofclass):
                if i not in [args.first, args.second]:
                    first_i += global_epoch_confusion[-1]["confusion"][(args.first, i)]
                    second_i += global_epoch_confusion[-1]["confusion"][(args.second, i)]

            print('obj1_count*first_i:', obj1_count*first_i)
            print('obj2_count*second_i:', obj2_count*second_i)

            print('obj1_count*first_second:', obj1_count*global_epoch_confusion[-1]["confusion"][(args.first, args.second)])
            print('obj2_count*second_first:', obj2_count*global_epoch_confusion[-1]["confusion"][(args.second, args.first)])

            confusion = global_epoch_confusion[-1]["confusion"]
            val_sorted = sorted({k:v for k,v in confusion.items() if v > 0}.items(), reverse=True, key=lambda x:x[1])
            print('val_sorted', val_sorted)
            print('\n'*3)

            balanced_confusion = {}
            for p in confusion:
                p2 = (p[1], p[0])
                if p not in balanced_confusion and p2 not in balanced_confusion:
                    balanced_confusion[p] = confusion[p]
                    if p2 in confusion:
                        balanced_confusion[p] += confusion[p2]

            val_sorted_balanced = sorted({k:v for k,v in balanced_confusion.items() if v > 0}.items(), reverse=True, key=lambda x:x[1])
            print('val_sorted_balanced', val_sorted_balanced)
            exit()

        elif args.checkmodel_mode == 'all':
            print(global_epoch_confusion[-1]["confusion"])
            import seaborn as sn
            import pandas as pd
            import matplotlib.pyplot as plt
            arr = [[0 for _ in range(numberofclass)] for _ in range(numberofclass)]
            for i in range(numberofclass):
                for j in range(numberofclass):
                    if (i, j) in global_epoch_confusion[-1]["confusion"]:
                        arr[i][j] = global_epoch_confusion[-1]["confusion"][(i, j)]
            cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            df_cm = pd.DataFrame(arr, index = [i for i in cifar10_labels], columns = [i for i in cifar10_labels])
            plt.figure(figsize = (10,7))
            sn.heatmap(df_cm, annot=True, vmin=0, vmax=0.15)
            plt.savefig('confusion_matrix.pdf')
            exit()
        else:
            print('invalid args.checkmodel_mode:', args.checkmodel_mode)
        exit()

    for epoch in range(0, args.epochs):
        global_epoch_confusion.append({})
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, target_train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, target_val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint

        if epoch >= 0: # save the best model througout the training process
            is_best = err1 <= best_err1
            best_err1 = min(err1, best_err1)
            if is_best:
                best_err5 = err5
                best_err1 = err1

            print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
            save_checkpoint({
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_err1': best_err1,
                'best_err5': best_err5,
                'optimizer': optimizer.state_dict(),
            }, is_best)


        get_confusion(val_loader, model, criterion, epoch)
        # cat->dog confusion
        log_print(str(args.first) + " -> " + str(args.second))
        log_print(global_epoch_confusion[-1]["confusion"][(args.first, args.second)])
        # dog->cat confusion
        log_print(str(args.second) + " -> " + str(args.first))
        log_print(global_epoch_confusion[-1]["confusion"][(args.second, args.first)])

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    epoch_confusions = 'runs/%s/' % (args.expname) + \
        'epoch_confusion_' + args.expid
    np.save(epoch_confusions, global_epoch_confusion)
    log_print("")
    # output best model accuracy and confusion
    repaired_model = 'runs/%s/' % (args.expname) + 'model_best.pth.tar'
    if os.path.isfile(repaired_model):
        print("=> loading checkpoint '{}'".format(repaired_model))
        checkpoint = torch.load(repaired_model)
        model.load_state_dict(checkpoint['state_dict'])
        top1err, _, _ = get_confusion(val_loader, model, criterion)
        # dog->cat confusion
        log_print(str(args.first) + " -> " + str(args.second))
        log_print(global_epoch_confusion[-1]["confusion"][(args.first, args.second)])
        # cat->dog confusion
        log_print(str(args.second) + " -> " + str(args.first))
        log_print(global_epoch_confusion[-1]["confusion"][(args.second, args.first)])

        accuracy = 100 - top1err
        v1 = global_epoch_confusion[-1]["confusion"][(args.first, args.second)]
        v2 = global_epoch_confusion[-1]["confusion"][(args.second, args.first)]
        v_avg = (v1+v2)/2

        performance_str = '%.2f_%.4f_%.4f_%.4f.txt' % (accuracy, v_avg, v1, v2)
        performance_file = os.path.join(directory, performance_str)
        with open(performance_file, 'w') as f_out:
            pass



def train(train_loader, target_train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    extra_iterator = iter(target_train_loader)
    t = tqdm(train_loader, desc = 'Train %d' % epoch)
    for i, (input, target) in enumerate(t):
        # measure data loading time
        data_time.update(time.time() - end)
        try:
            (target_input, target_target) = next(extra_iterator)
        except StopIteration:
            extra_iterator = iter(target_train_loader)
            (target_input, target_target) = next(extra_iterator)
        input = torch.cat([input, target_input])
        target = torch.cat([target, target_target])
        input = input.cuda()
        target = target.cuda()
        target_copy = target.cpu().numpy()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample

            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index,
                                                      :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                       (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a).mean() * lam + \
                criterion(output, target_b).mean() * (1. - lam)
            id3 = []
            id5 = []

            for j in range(len(input)):
                if (target_copy[j]) == args.first:
                    id3.append(j)
                elif (target_copy[j]) == args.second:
                    id5.append(j)

            m = nn.Softmax(dim=1)

            p_dist = torch.dist(torch.mean(
                m(output)[id3], 0), torch.mean(m(output)[id5], 0), 2)

            p_dist = 0
            loss2 = args.lam * loss - (1 - args.lam) * p_dist
        else:
            # compute output

            output = model(input)
            #target_output = model(target_input)
            #_, top1_output = output.max(1)
            #yhats = top1_output.cpu().data.numpy()
            # print(yhats[:5])
            if args.addition.contain("exp_newbn_weighted_loss"):
                def get_target_loss(target, output, ind1, ind2):
                    inds_first_1 = torch.where(target == ind1)
                    inds_first_2 = torch.where(torch.argmax(output, dim=1) == ind2)
                    inds_first = np.intersect1d(inds_first_1[0].cpu(), inds_first_2[0].cpu())
                    inds_first_cuda = torch.from_numpy(inds_first).cuda()

                    inds_second_1 = torch.where(target == ind2)
                    inds_second_2 = torch.where(torch.argmax(output, dim=1) == ind1)
                    inds_second = np.intersect1d(inds_second_1[0].cpu(), inds_second_2[0].cpu())
                    inds_second_cuda = torch.from_numpy(inds_second).cuda()

                    use_loss_target = False
                    loss_target = None
                    if len(inds_first) > 0 and len(inds_second) > 0:
                        loss_target = (criterion(output[inds_first], target[inds_first]).mean() + criterion(output[inds_second], target[inds_second]).mean()) / 2
                        use_loss_target = True
                    elif len(inds_first) > 0:
                        loss_target = criterion(output[inds_first], target[inds_first]).mean()
                        use_loss_target = True
                    elif len(inds_second) > 0:
                        loss_target = criterion(output[inds_second], target[inds_second]).mean()
                        use_loss_target = True
                    #print('len(inds_first), len(inds_second), loss_target.detach().numpy().cpu()', len(inds_first), len(inds_second), loss_target.detach().numpy().cpu())
                    return loss_target, use_loss_target

                if args.target_weight < 1:
                    target_weight = args.target_weight

                    loss_target, use_loss_target = get_target_loss(target, output, args.first, args.second)

                    if use_loss_target:
                        loss2 = target_weight * criterion(output, target).mean() +  (1-target_weight) * loss_target
                    else:
                        loss2 = criterion(output, target).mean()
                else:
                    loss2 = criterion(output, target).mean()
            else:

                id3 = []
                id5 = []
                for j in range(len(input)):
                    if (target_copy[j]) == args.first:
                        id3.append(j)
                    elif (target_copy[j]) == args.second:
                        id5.append(j)
                # print(output.shape)
                # print(output[id3].shape)
                # print((torch.sum(output[id3],0)/len(id3)).shape)
                m = nn.Softmax(dim=1)
                if len(id3) == 0 or len(id5) == 0:
                    p_dist = 0
                    print("not enough sample")
                    print(len(id3))
                    print(len(id5))
                else:
                    p_dist = torch.dist(torch.mean(
                        m(output)[id3], 0), torch.mean(m(output)[id5], 0), 2)

                loss = criterion(output, target).mean()
                loss2 = args.lam * loss - (1 - args.lam) * p_dist

        losses.update(loss2.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        t.set_postfix(loss = losses.avg)

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


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


def validate(val_loader, target_val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    t = tqdm(val_loader, desc = 'Val %d' % epoch)
    for i, (input, target) in enumerate(t):
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.mean().item(), input.size(0))
        t.set_postfix(loss = losses.avg)
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def get_confusion(val_loader, model, criterion, epoch=-1):
    global global_epoch_confusion
    global_epoch_confusion[-1]["epoch"] = epoch
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    correct = 0
    total = 0
    yhats = []
    labels = []

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)

        if "exp_newbn_softmax" in args.addition:
            eta = args.eta
            #chosen_ classes = torch.tensor([3, 5])
            #other_ classes = torch.tensor([0, 1, 2, 4, 6, 7, 8, 9])
            softmax = torch.nn.Softmax()
            output = softmax(output)
            output = output.detach().cpu().numpy()
            chosen_classes = np.array([args.first, args.second])
            other_classes = np.array([i for i in range(10) if i != args.first and i != args.second])
            #output[:, chosen_ classes] = torch.index_select(output, 0, chosen_ classes) - eta
            #output[:, non_chosen_ classes] = torch.index_select(output, 0, other_ classes) + eta
            output[:, chosen_classes] *= eta
            output = torch.from_numpy(output)
            output = torch.clamp(output, 0, 1).cuda()

            #output = softmax(output)
        _, top1_output = output.max(1)
        total += target.size(0)
        correct += top1_output.eq(target).sum().item()

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.mean().item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

        for i in range(len(input)):
            yhats.append(int(top1_output[i].cpu().data.numpy()))
            labels.append(int(target[i].cpu().data.numpy()))
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    acc = 100.*correct/total
    log_print(acc)

    correct = 0
    for i in range(len(labels)):
        if labels[i] == yhats[i]:
            correct += 1
    log_print(correct*1.0/len(labels))

    labels_list = []
    for i in range(10):
        labels_list.append(i)

    type1confusion = {}

    for l1 in labels_list:
        for l2 in labels_list:
            if l1 == l2 or ((l1, l2) in type1confusion):
                continue
            c = 0
            subcount = 0
            for i in range(len(yhats)):

                if l1 == labels[i] and l2 == yhats[i]:
                    c = c + 1

                if l1 == labels[i]:
                    subcount = subcount + 1

            type1confusion[(l1, l2)] = c*1.0/subcount
    global_epoch_confusion[-1]["confusion"] = type1confusion
    global_epoch_confusion[-1]["accuracy"] = acc

    dog_cat_sum = 0
    dog_cat_acc = 0
    for i in range(len(yhats)):

        if args.first == labels[i] or args.second == labels[i]:
            dog_cat_sum += 1
            if labels[i] == yhats[i]:
                dog_cat_acc += 1
    global_epoch_confusion[-1]["dogcatacc"] = dog_cat_acc/dog_cat_sum
    log_print("pair accuracy: " + str(global_epoch_confusion[-1]["dogcatacc"]))
    if args.save_npy:
        np.save(args.expname + '_yhats.npy', yhats)
        np.save(args.expname + '_labels.npy', labels)

    return top1.avg, top5.avg, losses.avg


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
