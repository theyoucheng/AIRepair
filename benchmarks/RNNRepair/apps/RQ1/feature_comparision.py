
import sys
# sys.path.append("../../")
import joblib
import numpy as np
import os
import argparse
import torch
import torch.nn as nn

from RNNRepair.utils import create_args,get_project_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')
    # parser.add_argument('-pca', default=10, type=int)
    # parser.add_argument('-epoch', default=30, type=int)
    #
    # parser.add_argument('-components', default=37, type=int)
    # parser.add_argument('-path')
    #
    # parser.add_argument('-model', default='torch_gru_toxic',
                        # choices=['keras_lstm_mnist', 'torch_gru_imdb', 'torch_gru_toxic', 'torch_lstm_bin', 'torch_gru_sst', ])
    args = create_args().parse_args()

    save_dir = get_project_path(args.path, args.model )

    if args.model == 'keras_lstm_mnist' or args.model == 'keras_gru_mnist':
        class_num = 10
    else:
        class_num =2

    pca_dir = os.path.join(save_dir, 'pca_trace')
    k = args.pca
    epoch = args.epoch
    k_path = os.path.join(pca_dir, str(k) + '_' + str(epoch) + '.ptr')
    pca_path = os.path.join(pca_dir, str(k) + '_' + str(epoch) + '.pca')
    pca = joblib.load(pca_path)


    pca_data, softmax, pred_seq_labels, pred_labels, train_labels = joblib.load(k_path)

    k_path = os.path.join(pca_dir, str(k) + '_' + str(epoch) + '.tptr')
    test_pca_data, test_softmax, test_seq_labels, test_pred_labels, test_truth_labels = joblib.load(k_path)

    print('Original RNN Test Acc:', np.sum(test_pred_labels == test_truth_labels)/len(test_truth_labels))
    # Correct training indexes
    indexes = np.where(pred_labels == train_labels)[0]
    test_indexes = np.where(test_pred_labels == test_truth_labels)[0]

    label = torch.from_numpy(train_labels[indexes]).long()


    test_labels = torch.from_numpy(test_truth_labels).long().to(device)

    abst_dir = os.path.join(save_dir, 'abs_model')
    log_path = os.path.join(abst_dir, str(args.pca) + '_' + str(args.epoch) +'_acc.log')
    plot_file = open(log_path, 'a+')

    stnum = args.components
    type = ['RL', 'ID', 'ID_RL', 'CS', 'RL_ID_CS']

    path = os.path.join(abst_dir, str(args.pca)+'_'+str(args.epoch)+'_'+str(stnum)+'_GMM.ast')

    if not os.path.exists(path):
        print('Cannot find the abstract model')
        exit()

    best_model = joblib.load(path)

    feature_path = os.path.join(save_dir, 'feature_data',
                                str(k) + '_' + str(epoch) + '_' + str(stnum) + '_feature.npz')
    if not os.path.exists(feature_path):
        print('Cannot find the feaures')
        exit()

    for t in type:
        train_trace, test_trace = joblib.load(feature_path)

        train_len = len(train_trace)
        train_trace.extend(test_trace)

        train_trace = [torch.from_numpy(i) for i in train_trace]

        total = torch.nn.utils.rnn.pad_sequence(train_trace, batch_first=True)

        # Preprocess
        total[:, :, 0] = total[:,:, 0] + 1
        total[:, :, 1] = total[:, :, 1] + 1
        total[:,:,0]/=(stnum+1)
        total[:,:,1]/=(class_num+1)

        if t == 'RL':
            total[:,:,2:]=0
            total[:,:,0] = 0
        elif t == 'ID':
            total[:, :, 1] = 0
            total[:, :, 2:] = 0
        elif t == 'ID_RL':
            total[:, :, 2:] = 0
        elif t == 'CS':
            total[:, :, 0:2] = 0
        else:
            print('Keep All')


        train_trace = total[0:train_len]
        test_trace = total[train_len:]

        input = train_trace[indexes].float()

        _, y, z = total.shape


        class my_m(nn.Module):
            def __init__(self):
                super(my_m, self).__init__()
                self.fc = nn.Linear(y * z, class_num)

            def forward(self, x):
                if len(x.shape) != 2:
                    x = x.reshape(x.shape[0], -1)
                return self.fc(x)


        epochs = 500
        model = my_m().to(device)
        learning_rate = 0.02
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


        loss_fn = nn.CrossEntropyLoss()
        label = label.to(device)
        data_inx = input.to(device)

        print(data_inx.shape)
        for idx in range(epochs):
            data_out = model(data_inx)

            loss = loss_fn(data_out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(data_out.data, -1)
            predicted = predicted.to(device)

        with torch.no_grad():
            test_inx = test_trace.float().to(device)
            test_out = model(test_inx)
            _, test_predicted = torch.max(test_out.data, -1)
            print(
                t,
                 loss.item(),
                 (predicted == label).sum().item() / len(label),
                 (test_labels == test_predicted).sum().item() / len(test_truth_labels)
                 )
