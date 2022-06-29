#python2 repair_confusion.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair
import math, os, random, json, pickle, sys, pdb
import string, shutil, time, argparse
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm as tqdm

import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch, torchvision
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from data_loader import CocoObject
from model import MultilabelObject
from itertools import cycle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../common"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../common/CutMix-PyTorch"))
from newbatchnorm2 import dnnrepair_BatchNorm2d

global_epoch_confusion = []
def main():
    global global_epoch_confusion
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='log' ,
                        help='path for saving trained models and log info')
    parser.add_argument('--ann_dir', type=str, default='/media/data/dataset/coco/annotations',
                        help='path for annotation json file')
    parser.add_argument('--image_dir', default = '/media/data/dataset/coco')


    parser.add_argument('--finetune', default=0, type=int)
    parser.add_argument('--num_epochs', type=int, default=18)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64) # batch size should be smaller if use text
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--first', default="person", type=str,
                        help='first object index')
    parser.add_argument('--second', default="bus", type=str,
                        help='second object index')
    parser.add_argument(
    '--pretrained', default='/set/your/model/path', type=str, metavar='PATH')
    parser.add_argument('--ratio', default=0.5, type=float,
                    help='target ratio for batchnorm layers')
    parser.add_argument('--replace', help='replace bn layer ',
                    action='store_true')
    parser.add_argument('--groupname', default="", type=str, help='output file name')
    args = parser.parse_args()
    assert os.path.isfile(args.pretrained)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    # Image preprocessing
    train_transform = transforms.Compose([
        transforms.Scale(args.image_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    val_transform = transforms.Compose([
        transforms.Scale(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])
    # Data samplers.
    train_data = CocoObject(ann_dir = args.ann_dir, image_dir = args.image_dir,
        split = 'train', transform = train_transform)
    val_data = CocoObject(ann_dir = args.ann_dir, image_dir = args.image_dir,
        split = 'val', transform = val_transform)


    # Data loaders / batch assemblers.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
                                              shuffle = True, num_workers = 4,
                                              pin_memory = False)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
                                            shuffle = False, num_workers = 4,
                                            pin_memory = False)
    # Build the models
    model = MultilabelObject(args, 80).cuda()


    criterion = nn.BCEWithLogitsLoss(weight = torch.FloatTensor(train_data.getObjectWeights()), size_average = True, reduction='None').cuda()

    def trainable_params():
        for param in model.parameters():
            if param.requires_grad:
                yield param

    optimizer = torch.optim.Adam(trainable_params(), args.learning_rate, weight_decay = 1e-5)

    if args.replace:
        model.to('cpu')
        global glob_bn_count
        global glob_bn_total
        glob_bn_total = 0
        glob_bn_count = 0
        count_bn_layer(model)
        print("total bn layer: " + str(glob_bn_total))
        glob_bn_count = 0
        replace_bn(model, args)
        print(model)
        model = model.cuda()


    best_performance = 0
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        args.start_epoch = checkpoint['epoch']
        best_performance = checkpoint['best_performance']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("no pre-trained model found")
        exit()

    #train_features_label_confusion = get_features(args, model, criterion, train_loader, optimizer, train_data)
    #np.save(os.path.join(args.groupname + '_train_data.npy'), train_features_label_confusion)
    test_features_label_confusion = get_features(args, model, criterion, val_loader, optimizer, val_data)
    np.save(os.path.join(args.groupname + '_test_data.npy'), test_features_label_confusion)


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


def replace_bn(module, args):
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
            replace_bn(child, args)

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

def save_checkpoint(args, state, is_best, filename):
    print("saving best model")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.log_dir, 'model_best_further.pth.tar'))


def compute_confusion(confusion_matrix, first, second):
    confusion = 0
    if (first, second) in confusion_matrix:
        confusion += confusion_matrix[(first, second)]

    if (second, first) in confusion_matrix:
        confusion += confusion_matrix[(second, first)]
    return confusion/2

def compute_bias(confusion_matrix, first, second, third):
    return abs(compute_confusion(confusion_matrix, first, second) - compute_confusion(confusion_matrix, first, third))

def get_features(args, model, criterion, val_loader, optimizer, test_data):
    model.eval()
    res = list()
    feature_data = {}
    features = []
    yhats = []
    labels = []
    image_ids = test_data.image_ids
    image_path_map = test_data.image_path_map
    #80 objects
    id2object = test_data.id2object
    id2labels = test_data.id2labels
    t = tqdm(val_loader)
    for batch_idx, (images, objects, image_ids) in enumerate(t):
        # Set mini-batch dataset
        images = Variable(images).cuda()
        objects = Variable(objects).cuda()

        object_preds = model(images)

        m = nn.Sigmoid()
        object_preds_r = m(object_preds)
        object_preds_c = object_preds_r.cpu().data.numpy()
        for i in range(len(image_ids)):
            yhat = []
            label = id2labels[image_ids.cpu().numpy()[i]]

            for j in range(len(object_preds[i])):
                a = object_preds_c[i][j]
                if a > 0.5:
                    yhat.append(id2object[j])
            yhats.append(yhat)
            labels.append(label)
        '''
        m = nn.Sigmoid()
        object_preds_r = m(object_preds)
        '''
        loss = criterion(object_preds, objects).mean()

        object_preds_max = object_preds.data.max(1, keepdim=True)[1]
        object_correct = torch.gather(objects.data, 1, object_preds_max).cpu().sum()
        res.append((image_ids, object_preds.data.cpu(), objects.data.cpu()))
        features.append(object_preds.data.cpu())


    # compute mean average precision score for object classifier
    preds_object   = torch.cat([entry[1] for entry in res], 0)
    targets_object = torch.cat([entry[2] for entry in res], 0)
    eval_score_object = average_precision_score(targets_object.numpy(), preds_object.numpy())
    print('\nmean average precision of object classifier on test data is {}\n'.format(eval_score_object))

    object_list = []
    for i in range(80):
        object_list.append(id2object[i])
    type2confusion = {}

    pair_count = {}
    confusion_count = {}
    type2confusion = {}


    for li, yi in zip(labels, yhats):
        no_objects = [id2object[i] for i in range(80) if id2object[i] not in li]
        for i in li:
            for j in no_objects:
                if (i, j) in pair_count:
                    pair_count[(i, j)] += 1
                else:
                    pair_count[(i, j)] = 1

                if i in yi and j in yi:
                    if (i, j) in confusion_count:
                        confusion_count[(i, j)] += 1
                    else:
                        confusion_count[(i, j)] = 1

    for i in object_list:
        for j in object_list:
            if i == j or (i, j) not in confusion_count or pair_count[(i, j)] < 10:
                continue
            type2confusion[(i, j)] = confusion_count[(i, j)]*1.0 / pair_count[(i, j)]
    feature_data["confusion"] = type2confusion
    feature_data["accuracy"] = eval_score_object
    feature_data["objects"] = object_list
    feature_data["labels"] = labels
    feature_data["preds"] = yhats
    feature_data["features"] = features
    return feature_data

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
    if epoch <= 16:
        lr = 0.0001
    else:
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    global_epoch_confusion[-1]["lr"] = lr

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

if __name__ == '__main__':
    main()
