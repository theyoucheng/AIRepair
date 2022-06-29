import os, sys
import copy
import numpy as np
import torch.nn as nn 
import joblib
import torch
import torch.nn.functional as F
from torchvision import transforms
# sys.path.append("../../")
from RNNRepair.use_cases.image_classification import MyNet
from RNNRepair.use_cases.image_classification.mnist_rnn_profile_torch import RNN
from sgd_train import sgd_train
import time
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# model structure
time_steps = 28
n_inputs = 28
hidden_size = 100
num_layers = 1
n_classes = 2 
model_type = "lstm"
### parameters ###
# for data
# nval = 10000

# for training in icml
batch_size = 1000
lr = 0.005
momentum = 0.9
num_epochs = 2
# for evaluation, the num of groups
test_batch_size = 4

# for storing
bundle_size = 15
### parameters ###

def preprocess( x_test):
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test

def compute_gradient(model, criterion, device, x_test,y_test, all_idx):
    n = all_idx.size
    if n > 40:
        test_batch_size = (n//10 + 1)
    else:
        test_batch_size = 4
    grad_idx = [item for item in np.array_split(np.arange(n), test_batch_size) if len(item)>0]
    u = [torch.zeros(*param.shape, requires_grad=False).to(device) for param in model.parameters()]
    model.train()
    for cur_idx in grad_idx:

        images = x_test[cur_idx]
        labels = y_test[cur_idx]
        images = torch.from_numpy(images.reshape(-1, time_steps, n_inputs)).to(device)
        labels = torch.from_numpy(labels).to(device)
        _, outputs = model(images)
        # print(Xtr.shape,ytr.shape,outputs.shape,F.softmax(outputs, dim=1).shape)
        loss = criterion(F.softmax(outputs, dim=1), labels)

        model.zero_grad()
        loss.backward()
        for j, param in enumerate(model.parameters()):
            u[j] += param.grad.data / n
    return u

def infl_sgd(output_path, target_epoch, flip_file ,test_file, id_ , seed=0 ):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)
    # setup # data split
    (x_train, y_train), (x_test, y_test), flip_idx = joblib.load(flip_file)
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    ntr = len(x_train)
    idx_train = np.arange(ntr)
    idx_test = np.load(test_file)
    nte = len(idx_test)
    print("[load,mode:{},seed:{}]load existed flip_idx file:{}.".format(flip,seed,flip_file))
    print("[load,seed:{}]train data:{},{}".format(seed,x_train.shape,y_train.shape))
    print("[load,seed:{}]target test data:{},{}".format(seed,x_test[idx_test].shape,y_test[idx_test].shape))

    # model setup
    torch.manual_seed(seed)
    model = RNN(n_inputs, hidden_size, num_layers, n_classes, model_type).to(device)
    criterion = nn.BCELoss()
    fn =\
                      '%s/epoch%02d_final_model.dat' % (output_path, target_epoch)

    model.load_state_dict(torch.load(fn))
    model.to(device)
    # model.eval()
    model.train()
    # gradient
    u = compute_gradient(model, criterion, device, x_test,y_test, idx_test)
    # model list
    list_of_models = [RNN(n_inputs, hidden_size, num_layers, n_classes, model_type).to(device) for _ in range(bundle_size)]
    models = MyNet.NetList(list_of_models)
    
    # influence
    origin_time=start_time = time.time()
    infl = torch.zeros(ntr, target_epoch, requires_grad=False).to(device)
    with torch.backends.cudnn.flags(enabled=False):
        for epoch in range(target_epoch, 0, -1):
            fn =\
                  '%s/epoch%02d_info.dat' % (output_path, epoch)
            info = joblib.load(fn)
            T = len(info)
            B = [bundle_size] * 3
            B.append(T % (bundle_size * 3))
            c = -1
            for k in range(3, -1, -1):
                fn = \
                          '%s/epoch%02d_bundled_models%02d.dat' % (output_path, epoch, k)

                models.load_state_dict(torch.load(fn))
                for t in range(B[k]-1, -1, -1):
                    m = models.models[t].to(device)
                    idx, seeds, lr = info[c]['idx'], info[c]['seeds'], info[c]['lr']
                    c -= 1
                    # influence
                    Xtr = []
                    ytr = []
                    for i, s in zip(idx, seeds):
                        Xtr.append(x_train[i:i+1])
                        ytr.append(y_train[i:i+1])
                        image = torch.from_numpy(x_train[i:i+1].reshape(-1, time_steps, n_inputs)).to(device)
                        label = torch.from_numpy(y_train[i:i+1]).to(device)
                        _, output = m(image)
                        # print(Xtr.shape,ytr.shape,outputs.shape,F.softmax(outputs, dim=1).shape)
                        loss = criterion(F.softmax(output, dim=1), label)
                        m.zero_grad()
                        loss.backward()
                        for j, param in enumerate(m.parameters()):
                            infl[i, epoch-1] += lr * (u[j].data * param.grad.data).sum() / idx.size
                    # update u
                    Xtr = torch.from_numpy(np.concatenate(Xtr,axis=0)).to(device)#torch.stack(Xtr).to(device)
                    ytr = torch.from_numpy(np.concatenate(ytr,axis=0)).to(device)
                    _, outputs = m(Xtr)
                    loss = criterion(F.softmax(outputs, dim=1), ytr)
                    grad_params = torch.autograd.grad(loss, m.parameters(), create_graph=True)
                    ug = 0
                    for uu, g in zip(u, grad_params):
                        ug += (uu * g).sum()
                    m.zero_grad()
                    ug.backward()
                    for j, param in enumerate(m.parameters()):
                        u[j] -= lr * param.grad.data / idx.size
            print("finish {}/{} epoch:{}".format(epoch,target_epoch,time.time()-start_time))
            start_time = time.time()
            # save
            fn = \
                  '%s/infl_sgd_at_epoch%02d_%02d.dat' % (output_path, target_epoch,id_)
            joblib.dump(infl.cpu().numpy(), fn, compress=9)
            if epoch > 1:
                infl[:, epoch-2] = infl[:, epoch-1].clone()
    print(">>>Total time:{}".format(time.time()-origin_time))


def infl_icml(output_path, target_epoch, alpha, flip_file ,test_file, id_,seed):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)
    # setup # data split
    (x_train, y_train), (x_test, y_test), flip_idx = joblib.load(flip_file)
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    
    ntr = len(x_train)
    idx_train = np.arange(ntr)
    idx_test = np.load(test_file)
    nte = len(idx_test)
    print("[load,mode:{},seed:{}]load existed flip_idx file:{}.".format(flip,seed,flip_file))
    print("[load,seed:{}]train data:{},{}".format(seed,x_train.shape,y_train.shape))
    print("[load,seed:{}]target test data:{},{}".format(seed,x_test[idx_test].shape,y_test[idx_test].shape))
    
    # model setup
    torch.manual_seed(seed)
    model = RNN(n_inputs, hidden_size, num_layers, n_classes, model_type).to(device)
    criterion = nn.BCELoss()
    fn ='%s/epoch%02d_final_model.dat' % (output_path, target_epoch)
    model.load_state_dict(torch.load(fn))
    model.to(device)
    model.train()

    # gradient
    u = compute_gradient(model, criterion, device, x_test,y_test, idx_test)
    
    # Hinv * u with SGD
    # seed_train = 0
    num_steps = int(np.ceil(ntr / batch_size))
    v = [torch.zeros(*param.shape, requires_grad=True, device=device) for param in model.parameters()]
    optimizer = torch.optim.SGD(v, lr=lr, momentum=momentum)
    loss_train = []
    origin_time=start_time = time.time()
    with torch.backends.cudnn.flags(enabled=False):
        for epoch in range(num_epochs):
            model.train()
            
            # training
            np.random.seed(epoch)
            idx = np.array_split(np.random.permutation(ntr), num_steps)
            for j, i in enumerate(idx):
                Xtr = []
                ytr = []
                for ii in i:
                    Xtr.append(x_train[ii:ii+1])
                    ytr.append(y_train[ii:ii+1])
                    
                Xtr = torch.from_numpy(np.concatenate(Xtr,axis=0)).to(device)#torch.stack(Xtr).to(device)
                ytr = torch.from_numpy(np.concatenate(ytr,axis=0)).to(device)
                _, outputs = model(Xtr)
                loss = criterion(F.softmax(outputs, dim=1), ytr)

                grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                vg = 0
                for vv, g in zip(v, grad_params):
                    vg += (vv * g).sum()
                model.zero_grad()
                vgrad_params = torch.autograd.grad(vg, model.parameters(), create_graph=True)
                loss_i = 0
                for vgp, vv, uu in zip(vgrad_params, v, u):
                    loss_i += 0.5 * (vgp * vv + alpha * vv * vv).sum() - (uu * vv).sum()
                optimizer.zero_grad()
                loss_i.backward()
                optimizer.step()
                loss_train.append(loss_i.item())
                # print(loss_i.item())
            print("finish {}/{} epoch:{}".format(epoch,target_epoch,time.time()-start_time))
            start_time = time.time()
    # save
    fn =\
                      '%s/loss_icml_at_epoch%02d_%02d.dat' % (output_path, target_epoch,id_)
    joblib.dump(np.array(loss_train), fn, compress=9)
    
    # influence
    origin_time=start_time = time.time()
    infl = np.zeros(ntr)
    for i in range(ntr):

        image = torch.from_numpy(x_train[i:i+1].reshape(-1, time_steps, n_inputs)).to(device)
        label = torch.from_numpy(y_train[i:i+1]).to(device)
        _, output = model(image)
        loss = criterion(F.softmax(output, dim=1), label)

        model.zero_grad()
        loss.backward()
        infl_i = 0
        for j, param in enumerate(model.parameters()):
            infl_i += (param.grad.data.cpu().numpy() * v[j].data.cpu().numpy()).sum()
        infl[i] = - infl_i / ntr
        
    # save
    fn = '%s/infl_icml_at_epoch%02d_%02d.dat' % (output_path, target_epoch, id_)

    joblib.dump(infl, fn, compress=9)
    print(">>>Total time:{}".format(time.time()-origin_time))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')
    parser.add_argument('-save_dir', default="./save", type=str)

    parser.add_argument('-target', default="sst_gru_sgd", type = str)
    parser.add_argument('-infl_type', default="icml", type = str)
    parser.add_argument('-target_epoch', default=1, type = int)
    parser.add_argument('-id', default=1, type = int)

    parser.add_argument('-start_seed', default=1000, type=int)
    parser.add_argument('-end_seed', default=1010, type=int)
    parser.add_argument('-flip', default=1, type = int)
    parser.add_argument('-ratio', default=0.3, type=float)

    parser.add_argument('-first', default=1, type = int)
    parser.add_argument('-second', default=1, type = int)

    args=parser.parse_args()
    save_dir = args.save_dir 

    target = args.target
    infl_type = args.infl_type
    target_epoch = args.target_epoch
    id_ = int(args.id)
    ratio = float(args.ratio)
    first = int(args.first)
    second = int(args.second)
    start_seed = int(args.start_seed)
    end_seed = int(args.end_seed)
    flip = int(args.flip)


    
    # infl_sgd(output_path, target_epoch, flip_file ,test_file, id_ , seed=0, )
    # target = sys.argv[1]
    # infl_type = sys.argv[2]
    # target_epoch = int(sys.argv[3])
    # id_ = int(sys.argv[6])

    # ratio = float(sys.argv[8])
    # first = int(sys.argv[9])
    # second = int(sys.argv[10])
    # start_seed = int(sys.argv[4])
    # end_seed = int(sys.argv[5])
    # flip = int(sys.argv[7])
    assert infl_type in ['sgd', 'icml']

    os.makedirs(os.path.join(save_dir,
                     "./{}_{}_{}_{}_{:.1f}/".format(target,first,second,flip,ratio,
        )), exist_ok=True )
    seeds = np.arange(start_seed,end_seed)
    time_log = open(
        os.path.join(save_dir,
                     "./{}_{}_{}_{}_{:.1f}/{}_time_{}_{}.log".format(target,first,second,flip,ratio,
        infl_type,start_seed,end_seed)
                     ),"w+")
    start_time = origin_time = time.time()
    for cur_seed in seeds:
        test_file =os.path.join(save_dir, "./dataflip_{}_{}_{}/target_test_{}_{}_{}_{}_{}.npy".format(
            first,second,ratio,first,second,flip,ratio,cur_seed)
            )
        assert os.path.exists(test_file),test_file
        flip_file =os.path.join(save_dir, "./dataflip_{}_{}_{}/torch_lstm_bin/{}_{}/model/flip{}_{}_{}.data".format(
            first,second,ratio,cur_seed,flip,first,second,flip)
            )
        assert os.path.exists(flip_file),flip_file

        output_path =os.path.join(save_dir,
           './%s_%d_%d_%d_%.1f/%s_%02d' % (target,first,second,flip,ratio,target,cur_seed)
           )
        if infl_type == 'sgd':
            infl_sgd(output_path, target_epoch, flip_file, test_file, id_, cur_seed)
        elif infl_type == 'icml':
            alpha = 0.1
            # output_path, target_epoch, alpha, flip_file ,test_file, id_,seed
            infl_icml( output_path, target_epoch, alpha, flip_file, test_file , id_, cur_seed)
        time_log.write("strategy:{}\t\tseed:{}\t\tused:{:.4f}s\n".format(infl_type,cur_seed,time.time()-start_time))
        time_log.flush()
        start_time = time.time()
    print("Total time for seeds {} : {:.4f}\n".format(seeds,time.time()-origin_time))
    time_log.write("Total time for seeds {} : {:.4f}\n".format(seeds,time.time()-origin_time))
    time_log.flush()
    time_log.close()

