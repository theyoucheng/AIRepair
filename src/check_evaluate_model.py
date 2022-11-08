import argparse
import os
import torch
import numpy as np
import torch.optim as optim
import json
import time
from torchvision import datasets, transforms
from models import MLP, MnistNet
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../common"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../common/CutMix-PyTorch"))
import resnet as RN
from sklearn.decomposition import PCA
sys.path.append('../../')
import time
import torch.onnx
import onnx
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn

use_cuda = torch.cuda.is_available()




parser = argparse.ArgumentParser(description='Train NN with constraints.')

parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, mnist, fmnist, and pyamidnet')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, mnist, fmnist, cifar100, and imagenet)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument(
    '--pretrained', default='/set/your/model/path', type=str, metavar='PATH')
parser.add_argument('--checkmodel', help='Check model accuracy',
                    action='store_true')
parser.add_argument('--first', default=3, type=int,
                    help='first object index')
parser.add_argument('--second', default=5, type=int,
                    help='second object index')
parser.add_argument('--third', default=2, type=int,
                    help='third object index')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
global_epoch_confusion = []

args = parser.parse_args()

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
pca = None

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
            target_train_dataset = get_dataset_from_specific_classes(target_train_dataset, args.first, args.second, args.third)
            target_test_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=transform_test)
            target_test_dataset = get_dataset_from_specific_classes(target_test_dataset, args.first, args.second, args.third)
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
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass, args.bottleneck)
    elif args.net_type == 'mobilenetv2':
        from mobilenetv2 import MobileNetV2
        model = MobileNetV2()
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
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
     # for checking pre-trained model accuracy and confusion
        
    if args.checkmodel:
        global_epoch_confusion.append({})
        top1err, _, _ = get_confusion(val_loader, model, criterion)
        confusion_matrix = global_epoch_confusion[-1]["confusion"]

        print(str((args.first, args.second, args.third)) + " triplet: " +
            str(compute_bias(confusion_matrix, args.first, args.second, args.third)))
        print(str((args.first, args.second)) + ": " + str(compute_confusion(confusion_matrix, args.first, args.second)))
        print(str((args.first, args.third)) + ": " + str(compute_confusion(confusion_matrix, args.first, args.third)))

        bias_dict = {}
        first, second = args.first, args.second
        for i in range(numberofclass):
            if i not in [first, second]:
                cur_bias = compute_bias(confusion_matrix, first, second, i)
                if cur_bias > 0:
                    bias_dict[(first, second, i)] = cur_bias
        val_sorted = sorted(bias_dict.items(), reverse=True, key=lambda x:x[1])
        print('val_sorted', val_sorted)

        exit()
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

    return top1.avg, top5.avg, losses.avg

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
def log_print(var):
    print("logging filter: " + str(var))
    
def compute_confusion(confusion_matrix, first, second):
    confusion = 0
    if (first, second) in confusion_matrix:
        confusion += confusion_matrix[(first, second)]

    if (second, first) in confusion_matrix:
        confusion += confusion_matrix[(second, first)]
    return confusion/2

def compute_bias(confusion_matrix, first, second, third):
    return abs(compute_confusion(confusion_matrix, first, second) - compute_confusion(confusion_matrix, first, third))



if __name__ == '__main__':
    main()
