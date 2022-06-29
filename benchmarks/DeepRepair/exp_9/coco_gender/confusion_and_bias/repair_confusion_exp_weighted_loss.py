#python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair --first "person" --second "bus" --ann_dir '../../../coco/annotations' --image_dir '../../../coco/' --weight 1 --target_weight 0.5 --class_num 80
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
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_loader import CocoObject
from model import MultilabelObject
from itertools import cycle

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
    parser.add_argument('--weight', default=1, type=float,
                    help='oversampling weight')
    parser.add_argument('--target_weight', default=1, type=float,
                help='target_weight')
    parser.add_argument('--class_num', default=81, type=int,
                help='81:coco_gender;80:coco')

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
    object2id = val_data.object2id

    first_id = object2id[args.first]
    second_id = object2id[args.second]

    weights = [1.0 if first_id in train_data.labels[i] or second_id in train_data.labels[i] else args.weight for i in range(len(train_data.labels)) ]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(train_data.labels))

    # Data loaders / batch assemblers.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
                                              shuffle = False, num_workers = 4,
                                              pin_memory = False, sampler=sampler)


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

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        global_epoch_confusion.append({})
        adjust_learning_rate(optimizer, epoch)
        train(args, epoch, model, criterion, train_loader, optimizer, train_F, score_F, train_data, object2id)
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

def train(args, epoch, model, criterion, train_loader, optimizer, train_F, score_F, train_data, object2id):
    id2labels = train_data.id2labels

    image_ids = train_data.image_ids
    image_path_map = train_data.image_path_map
    #80 objects

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_logger = AverageMeter()
    correct_logger = AverageMeter()

    res = list()
    end = time.time()

    t = tqdm(train_loader, desc = 'Train %d' % epoch)
    for batch_idx, (images, objects, image_ids) in enumerate(t):

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
        loss = criterion(object_preds, objects).mean()

        def sigmoid(z):
            return 1/(1 + np.exp(-z))

        if args.target_weight < 1:
            target_weight = args.target_weight


            objects_np = objects.detach().cpu().numpy()
            object_preds_np = object_preds.detach().cpu().numpy()
            objects_np = sigmoid(objects_np)
            object_preds_np = sigmoid(object_preds_np)

            id1 = object2id[args.first]
            id2 = object2id[args.second]

            inds = np.arange(objects_np.shape[0])
            inds_first = (objects_np[:,id1] > 0.5) & (objects_np[:, id2] <= 0.5) & (object_preds_np[:, id1] > 0.5) & (object_preds_np[:, id2] > 0.5)
            inds_first = inds[inds_first]
            inds_first_cuda = torch.from_numpy(inds_first).cuda()

            inds = np.arange(objects_np.shape[0])
            inds_second = (objects_np[:, id2] > 0.5) & (objects_np[:,id1] <= 0.5) & (object_preds_np[:, id2] > 0.5) & (object_preds_np[:, id1] > 0.5)
            inds_second = inds[inds_second]
            inds_second_cuda = torch.from_numpy(inds_second).cuda()

            # print()
            # print('objects_np[0]', objects_np[0])
            #
            # print('np.sum(objects_np[:,id1] > 0.5)', np.sum(objects_np[:,id1] > 0.5), 'np.sum(objects_np[:, id2] <= 0.5)', np.sum(objects_np[:, id2] <= 0.5), 'np.sum(object_preds_np[:, id1] > 0.5)', np.sum(object_preds_np[:, id1] > 0.5), 'np.sum(object_preds_np[:, id2] > 0.5)', np.sum(object_preds_np[:, id2] > 0.5))
            #
            # print('np.sum(objects_np[:, id2] > 0.5)', np.sum(objects_np[:, id2] > 0.5), 'np.sum(objects_np[:,id1] <= 0.5)', np.sum(objects_np[:,id1] <= 0.5), 'np.sum(object_preds_np[:, id2] > 0.5)', np.sum(object_preds_np[:, id2] > 0.5), 'np.sum(object_preds_np[:, id1] > 0.5)', np.sum(object_preds_np[:, id1] > 0.5))

            use_loss_target = False
            loss_target = None
            if len(inds_first) > 0:
                loss_target_1 = criterion(object_preds[inds_first_cuda], objects[inds_first_cuda]).mean()
                loss_target = loss_target_1
                use_loss_target = True
            if len(inds_second) > 0:
                loss_target_2 = criterion(object_preds[inds_second_cuda], objects[inds_second_cuda]).mean()
                loss_target = loss_target_2
                use_loss_target = True
            if len(inds_first) > 0 and len(inds_second) > 0:
                loss_target = (loss_target_1 + loss_target_2) / 2


            # print('len(inds_first)', len(inds_first), 'len(inds_second)', len(inds_second))

            if use_loss_target:
                loss2 = target_weight * loss + (1-target_weight) * loss_target

                # print('loss_target.detach().cpu().numpy()', loss_target.detach().cpu().numpy())
            else:
                loss2 = loss

            # print('loss.detach().cpu().numpy()', loss.detach().cpu().numpy())
        else:
            loss2 = loss

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
    #80 objects
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

    for i in object_list:
        for j in object_list:
            if i == j or (i, j) not in confusion_count or pair_count[(i, j)] < 10:
                continue
            type2confusion[(i, j)] = confusion_count[(i, j)]*1.0 / pair_count[(i, j)]
    global_epoch_confusion[-1]["confusion"] = type2confusion
    global_epoch_confusion[-1]["accuracy"] = eval_score_object

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
