import os, sys
import copy
import numpy as np
import joblib
import torch
import torch.nn as nn
from torchvision import transforms
import keras 
import time
# sys.path.append("../../")
import argparse

import torch.nn.functional as F

from RNNRepair.use_cases.image_classification.mnist_rnn_profile_torch import RNN
from RNNRepair.use_cases.image_classification import MyNet


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def preprocess( x_test):
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# for training
batch_size = 256
lr = 0.1
momentum = 0.0
num_epochs = 30
# model structure
time_steps = 28
n_inputs = 28
hidden_size = 100
num_layers = 1
n_classes = 2 
model_type = "lstm"
# for storing
bundle_size = 15

target = "mnist_rnn_sgd"

def sgd_train(flip,ratio,first,second,seed,target,device):
    torch.manual_seed(seed)
    output_path = os.path.join(save_dir,
                               './%s_%d_%d_%d_%.1f/%s_%02d' % (target,first,second,flip,ratio,target,seed)
                               )
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        assert False,"please check the existed folder:{}".format(output_path)
    fn =os.path.join(save_dir, "./dataflip_{}_{}_{}/torch_lstm_bin/{}_{}/model/flip{}_{}_{}.data".format(
        first,second,ratio,seed,flip,first,second,flip) )

    assert os.path.exists(fn),fn
    print("[load] load existed flip data:",fn)
    (x_train, y_train), (x_test, y_test), flip_idx = joblib.load(fn)
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    train_n = len(x_train)
    test_n = len(x_test)
    idx_train = np.arange(train_n)
    idx_test = np.arange(test_n)
    print("[mode:{},seed:{}]load existed flip_idx file:{}.".format(flip,seed,fn))
    print("[load,seed:{}]train data:{},{}".format(seed,x_train.shape,y_train.shape))
    print("[load,seed:{}]test data:{},{}".format(seed,x_test.shape,y_test.shape))
    # model setup
    model = RNN(n_inputs, hidden_size, num_layers, n_classes, model_type).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.BCELoss()
    # model list
    list_of_models = [RNN(n_inputs, hidden_size, num_layers, n_classes, model_type).to(device) for _ in range(bundle_size)]
    # training
    seed_id = 0
    score = []
    
    num_steps = int(np.ceil(train_n / batch_size))
    origin_time = start_time = time.time()

    for epoch in range(num_epochs):
        epoch += 1
        model.train()
        np.random.seed(epoch)
        idx_list = np.array_split(np.random.permutation(train_n), num_steps)
        train_loss, train_acc = 0, 0
        test_loss, test_acc = 0, 0
        total = 0
        info = []
        c = 0
        k = 0
        for i in range(num_steps):
            idx = idx_list[i]
    
            list_of_models[c].load_state_dict(copy.deepcopy(model.state_dict()))
            seeds = list(range(seed_id, seed_id+idx.size))
            info.append({'idx':idx, 'lr':lr, 'seeds':seeds})
            c += 1
            if c == bundle_size or i == len(idx_list) - 1:
                fn =\
                                  '%s/epoch%02d_bundled_models%02d.dat' % (output_path, epoch, k)

                models = MyNet.NetList(list_of_models)
                torch.save(models.state_dict(), fn)
                k += 1
                c = 0
            # sgd
            total += len(idx)
            images = x_train[idx]
            labels = y_train[idx]
            seed_id += idx.size

            images = torch.from_numpy(images.reshape(-1, time_steps, n_inputs)).to(device)
            labels = torch.from_numpy(labels).to(device)
            _, outputs = model(images)
            # print(Xtr.shape,ytr.shape,outputs.shape,F.softmax(outputs, dim=1).shape)
            loss = criterion(F.softmax(outputs, dim=1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (torch.sum(
                torch.argmax(outputs, dim=-1) == torch.argmax(labels, dim=1))).float()  / len(labels)
        train_acc /= num_steps
        print('Epoch [{}/{}], Step [{}/{}], Train size: {}, Loss: {:.4f},  Train ACC: {:.4f}'
                    .format(epoch, num_epochs, 0, num_steps, total, train_loss/num_steps, train_acc))
            # Test the model one batch

        fn =\
                  '%s/epoch%02d_final_model.dat' % (output_path, epoch)
        torch.save(model.state_dict(), fn)
        fn = \
              '%s/epoch%02d_final_optimizer.dat' % (output_path, epoch)
        torch.save(optimizer.state_dict(), fn)
        fn = \
              '%s/epoch%02d_info.dat' % (output_path, epoch)

        joblib.dump(info, fn, compress=9)

        # evaluation
        data = torch.from_numpy(x_test)
        with torch.no_grad():
            images = data.reshape(-1, time_steps, n_inputs).to(device)
            _, outputs = model(images)
            logits = F.softmax(outputs, dim=1)
            te_loss = criterion(logits, torch.from_numpy(y_test).to(device))

        logits = logits.cpu().numpy()
        re = np.argmax(logits, axis=1)
        correct = np.sum(re == np.argmax(y_test, axis=1))
        total = len(y_test)
        test_acc = np.around(100 * correct / total,4)
        print('Test Accuracy of the model on the {} test images: {} %'.format(len(y_test),test_acc))
        score.append(( train_loss/num_steps, te_loss.item(), train_acc, test_acc))
        print("[epoch {}, time]:{}".format(epoch,time.time()-start_time))
        start_time = time.time()

    print(">>>total time:{}".format(time.time()-origin_time))
    # save
    fn =\
          '%s/score.dat' % (output_path)

    joblib.dump(np.array(score), fn, compress=9)

# args
# CUDA_VISIBLE_DEVICES=1 python sgd_train.py 1 2 0 0.3 1 7
# CUDA_VISIBLE_DEVICES=1 python sgd_train.py 1 2 2 0.3 1 7

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')
    parser.add_argument('-save_dir', default="./save", type=str)

    parser.add_argument('-start_seed', default=1000, type=int)
    parser.add_argument('-end_seed', default=1010, type=int)
    parser.add_argument('-flip', default=1, type = int)
    parser.add_argument('-ratio', default=0.3, type=float)

    parser.add_argument('-first', default=1, type = int)
    parser.add_argument('-second', default=1, type = int)

    args=parser.parse_args()
    save_dir = args.save_dir 
    # start_seed = int(sys.argv[1])
    # end_seed = int(sys.argv[2])
    # flip = int(sys.argv[3])
    # ratio = float(sys.argv[4])
    # first = int(sys.argv[5])
    # second = int(sys.argv[6])

    start_seed = int(args.start_seed)
    end_seed = int(args.end_seed)
    flip = int(args.flip)
    ratio = float(args.ratio)
    first = int(args.first)
    second = int(args.second)

    seeds = np.arange(start_seed,end_seed)
    for seed in seeds:
        sgd_train(flip,ratio,first,second,seed,target,device)
        print(">>>>>Finish cur seed:{}!".format(seed))