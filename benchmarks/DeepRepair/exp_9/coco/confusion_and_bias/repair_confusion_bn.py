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

    parser.add_argument('--resume', default=1, type=int, help='whether to resume from log_dir if existent')
    parser.add_argument('--finetune', default=0, type=int)
    parser.add_argument('--num_epochs', type=int, default=18)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64) # batch size should be smaller if use text
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--lam', default=1, type=float,
                    help='hyperparameter lambda')
    parser.add_argument('--first', default="person", type=str,
                        help='first object index')
    parser.add_argument('--second', default="bus", type=str,
                        help='second object index')
    parser.add_argument(
    '--pretrained', default='/set/your/model/path', type=str, metavar='PATH')
    parser.add_argument('--debug', help='Check model accuracy',
    action='store_true')
    parser.add_argument('--ratio', default=0.5, type=float,
                    help='target ratio for batchnorm layers')
    parser.add_argument('--replace', help='replace bn layer ',
                    action='store_true')
    parser.add_argument('--class_num', default=80, type=int,
                help='81:coco_gender;80:coco')
    parser.add_argument('--checkmodel', help='Check model accuracy',
                    action='store_true')
    args = parser.parse_args()
    assert os.path.isfile(args.pretrained)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.log_dir) and not args.resume:
        print('Path {} exists! and not resuming'.format(args.log_dir))
        return
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    #save all parameters for training
    with open(os.path.join(args.log_dir, "arguments.log"), "a") as f:
        f.write(str(args)+'\n')

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
    first_data = CocoObject(ann_dir = args.ann_dir, image_dir = args.image_dir,
        split = 'train', transform = train_transform, filter=args.first)
    second_data = CocoObject(ann_dir = args.ann_dir, image_dir = args.image_dir,
        split = 'train', transform = train_transform, filter=args.second)


    # Data loaders / batch assemblers.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
                                              shuffle = True, num_workers = 4,
                                              pin_memory = False)

    first_loader = torch.utils.data.DataLoader(first_data, batch_size = args.batch_size/2,
                                              shuffle = True, num_workers = 4,
                                              pin_memory = False)
    second_loader = torch.utils.data.DataLoader(second_data, batch_size = args.batch_size/2,
                                              shuffle = True, num_workers = 4,
                                              pin_memory = False)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
                                            shuffle = False, num_workers = 4,
                                            pin_memory = False)
    # Build the models
    model = MultilabelObject(args, args.class_num).cuda()


    criterion = nn.BCEWithLogitsLoss(weight = torch.FloatTensor(train_data.getObjectWeights()), size_average = True, reduction='None').cuda()

    def trainable_params():
        for param in model.parameters():
            if param.requires_grad:
                yield param

    optimizer = torch.optim.Adam(trainable_params(), args.learning_rate, weight_decay = 1e-5)


    if os.path.isfile(args.pretrained):
        train_F = open(os.path.join(args.log_dir, 'train.csv'), 'w')
        val_F = open(os.path.join(args.log_dir, 'val.csv'), 'w')
        score_F = open(os.path.join(args.log_dir, 'score.csv'), 'w')
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        args.start_epoch = checkpoint['epoch']
        best_performance = checkpoint['best_performance']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        exit()
    best_performance = 0

    if args.replace:
        model.to('cpu')
        global glob_bn_count
        global glob_bn_total
        glob_bn_total = 0
        glob_bn_count = 0
        count_bn_layer(model)
        print("total bn layer: " + str(glob_bn_total))
        glob_bn_count = 0
        replace_bn(model, args.ratio)
        print(model)
        model = model.cuda()

    if args.checkmodel:
        global_epoch_confusion.append({})
        epoch = 0
        _ = get_confusion(args, epoch, model, criterion, val_loader, optimizer, val_F, score_F, val_data)
        # obj1_count = global_epoch_confusion[-1]["obj1_count"]
        # obj2_count = global_epoch_confusion[-1]["obj2_count"]
        type2confusion = global_epoch_confusion[-1]["confusion"]
        pair_count = global_epoch_confusion[-1]["pair_count"]

        first_i_list = []
        second_i_list = []
        for (i, j) in type2confusion:
            if i == args.first:
                if j != args.second:
                    first_i_list.append(type2confusion[(i, j)] * pair_count[(i, j)])
            if i == args.second:
                if j != args.first:
                    second_i_list.append(type2confusion[(i, j)] * pair_count[(i, j)])

        first_i = np.sum(first_i_list)
        second_i = np.sum(second_i_list)

        first_i_max = np.max(first_i_list)
        second_i_max = np.max(second_i_list)

        print('first_i:', first_i)
        print('second_i:', second_i)
        print('first_i_max:', first_i_max)
        print('second_i_max:', second_i_max)
        if (args.first, args.second) in type2confusion:
            print('first_second confusion:', type2confusion[(args.first, args.second)])
            print('first_second:', type2confusion[(args.first, args.second)] * pair_count[(args.first, args.second)])
        if (args.second, args.first) in type2confusion:
            print('second_first confusion:', type2confusion[(args.second, args.first)])
            print('second_first:', type2confusion[(args.second, args.first)] * pair_count[(args.second, args.first)])

        confusion = type2confusion
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


    for epoch in range(args.start_epoch, args.num_epochs + 1):
        global_epoch_confusion.append({})
        adjust_learning_rate(optimizer, epoch)
        train(args, epoch, model, criterion, train_loader, optimizer, train_F, score_F, train_data, first_loader, second_loader)
        current_performance = get_confusion(args, epoch, model, criterion, val_loader, optimizer, val_F, score_F, val_data)
        is_best = current_performance > best_performance
        best_performance = max(current_performance, best_performance)
        model_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_performance': best_performance}
        save_checkpoint(args, model_state, is_best, os.path.join(args.log_dir, 'checkpoint.pth.tar'))
        confusion_matrix = global_epoch_confusion[-1]["confusion"]
        first_second = compute_confusion(confusion_matrix, args.first, args.second)

        accuracy = best_performance
        v_conf = first_second
        directory = args.log_dir
        performance_str = '%.4f_%.4f.txt' % (accuracy, v_conf)
        performance_file = os.path.join(directory, performance_str)
        if is_best:
            with open(performance_file, 'w') as f_out:
                pass
        #os.system('python plot.py {} &'.format(args.log_dir))

    train_F.close()
    val_F.close()
    score_F.close()
    np.save(os.path.join(args.log_dir, 'global_epoch_confusion.npy'), global_epoch_confusion)


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


def replace_bn(module, ratio):
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
                new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, ratio, child.eps, child.momentum, child.affine, track_running_stats=True)
                setattr(module, child_name, new_bn)
            else:
                print('replaced: bn')
                new_bn = dnnrepair_BatchNorm2d(child.num_features, child.weight, child.bias, child.running_mean, child.running_var, 1, child.eps, child.momentum, child.affine, track_running_stats=True)
                setattr(module, child_name, new_bn)
        else:
            replace_bn(child, ratio)

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

def train(args, epoch, model, criterion, train_loader, optimizer, train_F, score_F, train_data, first_loader, second_loader):
    id2labels = train_data.id2labels

    image_ids = train_data.image_ids
    image_path_map = train_data.image_path_map
    #args.class_num objects

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_logger = AverageMeter()
    correct_logger = AverageMeter()

    res = list()
    end = time.time()
    first_iterator = iter(first_loader)
    second_iterator = iter(second_loader)

    t = tqdm(train_loader, desc = 'Train %d' % epoch)
    for batch_idx, (images, objects, image_ids) in enumerate(t):
        try:
            (images1, objects1, image_ids1) = next(first_iterator)
        except StopIteration:
            first_iterator = iter(first_loader)
            (images1, objects1, image_ids1) = next(first_iterator)

        try:
            (images2, objects2, image_ids2) = next(second_iterator)
        except StopIteration:
            second_iterator = iter(second_loader)
            (images2, objects2, image_ids2) = next(second_iterator)

        images = torch.cat([images, images1, images2])
        objects = torch.cat([objects, objects1, objects2])
        image_ids = torch.cat([image_ids, image_ids1, image_ids2])
        # if batch_idx == 100: break # constrain epoch size
        labels = []
        for i in range(len(image_ids)):
            yhat = []
            label = id2labels[image_ids.cpu().numpy()[i]]
            labels.append(label)

        data_time.update(time.time() - end)
        # Set mini-batch dataset
        images = Variable(images).cuda()
        objects = Variable(objects).cuda()
        # Forward, Backward and Optimize
        optimizer.zero_grad()

        object_preds = model(images)
        if args.replace:
            loss = criterion(object_preds[:object_preds.size(
                0) // 2], objects[:objects.size(0) // 2]).mean()
        else:
            loss = criterion(object_preds, objects).mean()

        #loss = criterion(object_preds, objects).mean()
        '''
        m = nn.Softmax(dim=1)
        firstid = []
        secondid = []
        thirdid = []

        for j in range(len(labels)):
            if args.first in (labels[j]):# and "bus" not in (labels[j]):
                firstid.append(j)
            if args.second in (labels[j]):# and "person" not in (labels[j]):
                secondid.append(j)
        #print(len(labels))
        #print(object_preds.shape)
        if args.debug:
            print(len(firstid))
            print(len(secondid))

        if len(firstid) == 0 or len(secondid) == 0:
            p_dist2 = 0
            print("not enough sample")
        else:
            p_dist2 = torch.dist(torch.mean(m(object_preds)[firstid],0), torch.mean(m(object_preds)[secondid],0),2)
        '''
        loss2 = loss #- args.lam * p_dist2
        loss_logger.update(loss2.item())
        object_preds_max = object_preds.data.max(1, keepdim=True)[1]
        object_correct = torch.gather(objects.data, 1, object_preds_max).cpu().sum()
        correct_logger.update(object_correct)

        res.append((image_ids, object_preds.data.cpu(), objects.data.cpu()))

        loss2.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        t.set_postfix(loss = loss_logger.avg)

        train_F.write('{},{},{}\n'.format(epoch, loss.item(), object_correct))
        train_F.flush()

    # compute mean average precision score for object classifier
    preds_object   = torch.cat([entry[1] for entry in res], 0)
    targets_object = torch.cat([entry[2] for entry in res], 0)
    eval_score_object = average_precision_score(targets_object.numpy(), preds_object.numpy())
    print('\nmean average precision of object classifier on training data is {}\n'.format(eval_score_object))
    score_F.write('{},{},{}\n'.format(epoch, 'train', eval_score_object))
    score_F.flush()


def get_confusion(args, epoch, model, criterion, val_loader, optimizer, val_F, score_F, test_data):
    global global_epoch_confusion
    global_epoch_confusion[-1]["epoch"] = epoch
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_logger = AverageMeter()
    correct_logger = AverageMeter()

    res = list()
    end = time.time()
    yhats = []
    labels = []
    image_ids = test_data.image_ids
    image_path_map = test_data.image_path_map
    #args.class_num objects
    id2object = test_data.id2object
    id2labels = test_data.id2labels
    t = tqdm(val_loader, desc = 'Test %d' % epoch)
    for batch_idx, (images, objects, image_ids) in enumerate(t):
        # if batch_idx == 100: break # constrain epoch size

        data_time.update(time.time() - end)
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
        loss_logger.update(loss.item())

        object_preds_max = object_preds.data.max(1, keepdim=True)[1]
        object_correct = torch.gather(objects.data, 1, object_preds_max).cpu().sum()
        correct_logger.update(object_correct)

        res.append((image_ids, object_preds.data.cpu(), objects.data.cpu()))

        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        t.set_postfix(loss = loss_logger.avg)

    val_F.write('{},{},{}\n'.format(epoch, loss_logger.avg, correct_logger.avg))
    val_F.flush()

    # compute mean average precision score for object classifier
    preds_object   = torch.cat([entry[1] for entry in res], 0)
    targets_object = torch.cat([entry[2] for entry in res], 0)
    eval_score_object = average_precision_score(targets_object.numpy(), preds_object.numpy())
    print('\nmean average precision of object classifier on test data is {}\n'.format(eval_score_object))
    score_F.write('{},{},{}\n'.format(epoch, 'test', eval_score_object))
    score_F.flush()

    object_list = []
    for i in range(args.class_num):
        object_list.append(id2object[i])
    type2confusion = {}

    pair_count = {}
    confusion_count = {}
    type2confusion = {}

    obj1_count = 0
    obj2_count = 0
    for li, yi in zip(labels, yhats):
        no_objects = [id2object[i] for i in range(args.class_num) if id2object[i] not in li]
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
            if i == args.first:
                obj1_count += 1
            elif i == args.second:
                obj2_count += 1

    for i in object_list:
        for j in object_list:
            if i == j or (i, j) not in confusion_count or pair_count[(i, j)] < 10:
                continue
            type2confusion[(i, j)] = confusion_count[(i, j)]*1.0 / pair_count[(i, j)]
    global_epoch_confusion[-1]["confusion"] = type2confusion
    global_epoch_confusion[-1]["accuracy"] = eval_score_object

    global_epoch_confusion[-1]["pair_count"] = pair_count
    global_epoch_confusion[-1]["obj1_count"] = obj1_count
    global_epoch_confusion[-1]["obj2_count"] = obj2_count

    return eval_score_object

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
