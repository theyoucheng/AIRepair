
import sys
# sys.path.append("../../")
import joblib
import numpy as np
import os
import argparse
import torch
import torch.nn as nn

from RNNRepair.utils import get_project_path,create_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    args = create_args().parse_args()

    save_dir = get_project_path(args.path, args.model )
    
    if args.model == 'keras_lstm_mnist':
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

    indexes = np.where(pred_labels == train_labels)[0]
    test_indexes = np.where(test_pred_labels == test_truth_labels)[0]

    label = torch.from_numpy(train_labels[indexes]).long()


    test_labels = torch.from_numpy(test_truth_labels).long().to(device)

    abst_dir = os.path.join(save_dir, 'abs_model')
    log_path = os.path.join(abst_dir, str(args.pca) + '_' + str(args.epoch) +'_acc.log')
    plot_file = open(log_path, 'a+')


    for stnum in range(args.start, args.end, 1):
        print('Start train trace for', stnum)
        path = os.path.join(abst_dir, str(args.pca)+'_'+str(args.epoch)+'_'+str(stnum)+'_GMM.ast')

        if not os.path.exists(path):
            continue
        best_model = joblib.load(path)

        gan_train = os.path.join(save_dir, 'feature_data',
                                 str(k) + '_' + str(epoch) + '_' + str(stnum) + '_feature.npz')
        if not os.path.exists(gan_train):
            continue

        train_trace, test_trace = joblib.load(gan_train)

        train_len = len(train_trace)
        train_trace.extend(test_trace)

        train_trace = [torch.from_numpy(i) for i in train_trace]

        total = torch.nn.utils.rnn.pad_sequence(train_trace, batch_first=True)
        total[:, :, 0] = total[:,:, 0] + 1
        total[:, :, 1] = total[:, :, 1] + 1


        total[:,:,0]/=(stnum+1)
        total[:,:,1]/=(class_num+1)
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

            plot_file.write(
                "%d,%f,%f,%f\n" %
                (stnum,
                 loss.item(),
                 (predicted == label).sum().item() / len(label),
                 (test_labels == test_predicted).sum().item() / len(test_truth_labels)
                 ))

            print(
                stnum,
                 loss.item(),
                 (predicted == label).sum().item() / len(label),
                 (test_labels == test_predicted).sum().item() / len(test_truth_labels)
                 )
            plot_file.flush()
    plot_file.close()