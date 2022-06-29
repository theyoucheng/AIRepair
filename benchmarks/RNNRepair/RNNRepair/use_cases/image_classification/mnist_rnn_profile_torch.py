import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# import sys
# sys.path.append("../../")
from ...profile_abs import  Profiling
import os
import torch.utils.data as data
# Device configuration
import torch.optim as optim
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import joblib
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

import torch.nn.functional as F

# Recurrent neural network (many-to-one)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, rnn_type):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
#        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        if self.rnn_type == 'lstm':
            state_vec, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        else:
            state_vec, _ = self.rnn(x, h0)


        # Decode the hidden state of the last time step
        out = self.fc(state_vec[:, -1, :])
        # dense = self.fc(state_vec)
        return state_vec, out


'''
data : n * 28 * 28  batch_size 

'''
# from os.path import expanduser
# mnist_dir = os.path.join(expanduser('~') , '.torch')

class TorchMnistiClassifier(Profiling):
    def __init__(self, rnn_type, save_dir,train_default = True, epoch=10, seed = 0, flip = 0, first = 1, second = 7, ratio = 0.1):

        # # Hyper-parameters
        # sequence_length = 28
        # input_size = 28
        # hidden_size = 128
        #
        # num_layers = 2
        # num_classes = 10
        # batch_size = 100
        # num_epochs = 2


        super().__init__(save_dir)
        self.time_steps = 28  # timesteps to unroll
        self.n_units = 100  # hidden LSTM units
        self.n_inputs = 28  # rows of 28 pixels (an mnist img is 28x28)
        self.n_classes = 2  # mnist classes/labels (0-9)
        self.batch_size = 256  # Size of each batch
        self.num_layers = 1 # Number of recurrent layers.
        self.channel = 1
        self.n_epochs = epoch
        self.learning_rate = 0.1
        self.seed = seed
        self.ratio = ratio



        # Internal
        self._data_loaded = False
        self._trained = False
        self._rnn_type = rnn_type

        self.flip_first = first
        self.flip_second = second
        self.flip = flip



        if train_default:
            if self.flip > 0:
                self.model_path = os.path.join(self.model_dir, rnn_type + '_' + str(epoch)+'_'+ str(self.flip_first) +'_'+ str(self.flip_second) +'_'+str(self.flip)+ '.ckpt')
            else:
                self.model_path = os.path.join(self.model_dir,
                                               rnn_type + '_' + str(epoch) + '.ckpt')

            self.train(is_load=True, saved_path=self.model_path)
        #
        #
        # self.model = RNN(self.n_inputs, self.n_units, self.num_layers, self.n_classes, self._rnn_type).to(device)
        # if os.path.exists(self.model_path):
        #     print('Load existing model')
        #     self.model.load_state_dict( torch.load(os.path.join(self.model_path)))
        # else:
        #     print('Training...')
        #     print(device)
        #     self.train()

    def load_binary_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        name = 'flip' + str(self.flip_first) +'_'+ str(self.flip_second) +'_'+str(self.flip)+'.data'
        file_path = os.path.join(self.model_dir, name)
        np.random.seed(self.seed)
        if not os.path.exists(file_path):
            np.random.seed(self.seed)
            first_idx = np.where(y_train == self.flip_first)[0]
            second_idx = np.where(y_train == self.flip_second)[0]
            y_train[first_idx] = 0
            y_train[second_idx] = 1

            test_first_idx = np.where(y_test == self.flip_first)[0]
            test_second_idx = np.where(y_test == self.flip_second)[0]
            y_test[test_first_idx] = 0
            y_test[test_second_idx] = 1

            new_train_idx = np.concatenate([first_idx, second_idx])
            np.random.shuffle(new_train_idx)
            new_test_idx = np.concatenate([test_first_idx,test_second_idx])
            new_xtrain = x_train[new_train_idx]
            new_ytrain = y_train[new_train_idx]
            new_xtest = x_test[new_test_idx]
            new_ytest = y_test[new_test_idx]

            if self.flip == 1: #0 -> 1
                candidate_idx = np.where(new_ytrain == 0)[0]
                flip_idx = np.random.choice(candidate_idx,int(len(candidate_idx) * self.ratio), replace=False)
            elif self.flip == 2: # 1 -> 0
                candidate_idx = np.where(new_ytrain == 1)[0]
                flip_idx = np.random.choice(candidate_idx, int(len(candidate_idx) * self.ratio), replace=False)
            elif self.flip == 3:
                candidate_idx = np.arange(len(new_ytrain))
                flip_idx = np.random.choice(candidate_idx, int(len(candidate_idx) * self.ratio), replace=False)
            else:
                flip_idx = []

            new_ytrain[flip_idx] = 1 - new_ytrain[flip_idx]


            new_ytrain = to_categorical(new_ytrain)
            new_ytest = to_categorical(new_ytest)

            joblib.dump(((new_xtrain, new_ytrain), (new_xtest, new_ytest), flip_idx), file_path)
            return (new_xtrain, new_ytrain), (new_xtest, new_ytest), flip_idx
        else:
            return joblib.load(file_path)
    def retrain(self, removed_idx=None, is_load = False, saved_path = None, restore_idx=None,target_test_idx=None):
        torch.manual_seed(self.seed)
        model = RNN(self.n_inputs, self.n_units, self.num_layers, self.n_classes, self._rnn_type).to(device)
        if is_load == True and os.path.exists(saved_path):
            model.load_state_dict(torch.load(saved_path, map_location=device))
            model.to(device)

        else:
            import torch.nn.functional as F
            from tensorflow import keras
            (x_train, y_train), (x_test, y_test), _ = self.load_binary_data()
            if restore_idx is not None:
                y_train[restore_idx] = 1 - y_train[restore_idx]

            
            
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.0)
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            # Loss and optimizer
            criterion = nn.BCELoss()
            n_tr = len(x_train)
            num_steps = int(np.ceil(n_tr / self.batch_size))

            x_train = self.preprocess(x_train)
            x_test = self.preprocess(x_test)

            for epoch in range(self.n_epochs):
                epoch += 1
                model.train()
                np.random.seed(epoch)
                #random shuffle
                idx_list = np.array_split(np.random.permutation(n_tr), num_steps)
                train_loss, train_acc = 0, 0
                total = 0
                for i in range(num_steps):
                    idx = idx_list[i]
                    b = idx.size
                    if removed_idx is not None:
                        idx = np.setdiff1d(idx, removed_idx)

                    images = x_train[idx]
                    labels = y_train[idx]
                    labels = torch.from_numpy(labels).to(device)
                    data = torch.from_numpy(images)
                    images = data.reshape(-1, self.time_steps, self.n_inputs).to(device)

                    total += len(idx)
                    # Forward pass
                    _, outputs = model(images)
                    loss = criterion(F.softmax(outputs, dim=1), labels)
                    # loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    # for p in model.parameters():
                    #     p.grad.data *= idx.size / b
                    optimizer.step()

                    train_loss += loss.item()


                    train_acc += (torch.sum(
                        torch.argmax(outputs, dim=-1) == torch.argmax(labels, dim=1))).float() / len(labels)

                    # train_acc += (torch.sum(torch.argmax(outputs, dim=-1)==labels)).float()/len(labels)
                print('Epoch [{}/{}], Step [{}/{}], Train size: {}, Loss: {:.4f},  Train ACC: {:.4f}'
                      .format(epoch, self.n_epochs, 0, num_steps, total, train_loss/num_steps, train_acc/num_steps))
                # Test the model one batch
                data = torch.from_numpy(x_test)
                with torch.no_grad():
                    images = data.reshape(-1, self.time_steps, self.n_inputs).to(device)
                    state_vec, _ = model(images)
                    dense = model.fc(state_vec)
                    sv_softmax = torch.nn.functional.softmax(dense, dim=-1)
                sv_softmax = sv_softmax.cpu().numpy()
                re = np.argmax(sv_softmax, axis=-1)[:, -1]
                correct = np.sum(re == np.argmax(y_test, axis=1))
                total = len(y_test)
                test_acc = np.around(100 * correct / total,4)
                print('Test Accuracy of the model on the {} test images: {} %'.format(len(y_test),test_acc))

                if target_test_idx is not None:
                    target_correct = np.sum(re[target_test_idx] == np.argmax(y_test[target_test_idx], axis=1))
                    target_acc = np.around(100 * target_correct / len(target_test_idx),4)
                    print('Target Test Accuracy of the model on the {} test images: {} %'.format(len(target_test_idx),target_acc))

            if saved_path is not None:
                torch.save(model.state_dict(), saved_path)
            
        if target_test_idx is not None:
            return model,test_acc,target_acc
        else:
            self.model = model
            return model
    def train(self, flip_idx=None, is_load = False, saved_path = None):
        torch.manual_seed(self.seed)
        model = RNN(self.n_inputs, self.n_units, self.num_layers, self.n_classes, self._rnn_type).to(device)
        if is_load == True and os.path.exists(saved_path):
            model.load_state_dict(torch.load(saved_path, map_location=device))
            model.to(device)
        else:

            (x_train, y_train), (x_test, y_test), _ = self.load_binary_data()


            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.0)
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            # Loss and optimizer
            criterion = nn.BCELoss()
            n_tr = len(x_train)
            num_steps = int(np.ceil(n_tr / self.batch_size))

            x_train = self.preprocess(x_train)
            x_test = self.preprocess(x_test)
            if flip_idx is not None:
                y_train[flip_idx] = 1 - y_train[flip_idx]
            for epoch in range(self.n_epochs):
                epoch += 1
                model.train()
                np.random.seed(epoch)
                #random shuffle
                idx_list = np.array_split(np.random.permutation(n_tr), num_steps)
                train_loss, train_acc = 0, 0
                total = 0
                for i in range(num_steps):
                    idx = idx_list[i]
                    b = idx.size
                        # idx = np.setdiff1d(idx, removed_idx)
                    images = x_train[idx]
                    labels = y_train[idx]
                    labels = torch.from_numpy(labels).to(device)
                    data = torch.from_numpy(images)
                    images = data.reshape(-1, self.time_steps, self.n_inputs).to(device)

                    total += len(idx)
                    # Forward pass
                    _, outputs = model(images)

                    loss = criterion(F.softmax(outputs, dim=1), labels)
                    # loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    for p in model.parameters():
                        p.grad.data *= idx.size / b
                    optimizer.step()

                    train_loss += loss.item()


                    train_acc += (torch.sum(
                        torch.argmax(outputs, dim=-1) == torch.argmax(labels, dim=1))).float() / len(labels)

                    # train_acc += (torch.sum(torch.argmax(outputs, dim=-1)==labels)).float()/len(labels)
                print('Epoch [{}/{}], Step [{}/{}], Train size: {}, Loss: {:.4f},  Train ACC: {:.4f}'
                      .format(epoch, self.n_epochs, 0, num_steps, total, train_loss/num_steps, train_acc/num_steps))
                # Test the model one batch
                data = torch.from_numpy(x_test)
                with torch.no_grad():
                    images = data.reshape(-1, self.time_steps, self.n_inputs).to(device)
                    state_vec, _ = model(images)
                    dense = model.fc(state_vec)
                    sv_softmax = torch.nn.functional.softmax(dense, dim=-1)
                sv_softmax = sv_softmax.cpu().numpy()
                re = np.argmax(sv_softmax, axis=-1)[:, -1]
                correct = np.sum(re == np.argmax(y_test, axis=1))
                total = len(y_test)
                print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
            if saved_path is not None:
                torch.save(model.state_dict(), saved_path)
        self.model = model
        return model
        # torch.save(self.model.state_dict(), self.model_path)

    def do_profile(self, test=False):
        # train_dataset = torchvision.datasets.MNIST(root=mnist_dir,
        #                                            train=True,
        #                                            transform=transforms.ToTensor(),
        #                                            download=True)
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        #                                            batch_size=self.batch_size,
        #                                            shuffle=True)
        # return self.predict_loader(train_loader)
        # from tensorflow import keras

        # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        (x_train, y_train), (x_test, y_test),_ = self.load_binary_data()
        if test:
            # return self.predict(x_test, y_test)
            return self.predict(x_test, np.argmax(y_test, axis=1))
        else:
            # return self.predict(x_train, y_train)
            return self.predict(x_train, np.argmax(y_train, axis=1))

    def cal_loss(self, test_data, y, individual=True):
        criterion = nn.CrossEntropyLoss()
        test_data = self.preprocess(test_data)
        labels = torch.from_numpy(y).long().to(device)
        data = torch.from_numpy(test_data)
        images = data.reshape(-1, self.time_steps, self.n_inputs).to(device)

        _, outputs = self.model(images)

        pre_lal = torch.argmax(outputs, dim=-1).cpu().numpy()

        if individual:
            re = []
            for i, _ in enumerate(outputs):
                loss = criterion(outputs[i:i+1], labels[i:i+1])
                re.append(loss.item())
            return re, pre_lal
        else:
            return criterion(outputs, labels).item(), pre_lal


    def preprocess(self, x_test):
        # x_test = x_test.reshape(x_test.shape[0], self.time_steps, self.n_inputs, self.channel)
        x_test = x_test.astype('float32')
        x_test /= 255
        return x_test
    def predict_loader(self, data_loader):
        with torch.no_grad():
            t1,t2 = [],[]
            for images, labels in data_loader:
                images = images.reshape(-1, self.time_steps, self.n_inputs).to(device)

                state_vec, _ = self.model(images)
                dense = self.model.fc(state_vec)
                softmax = torch.nn.functional.softmax(dense, dim=-1)
                t1.append(softmax.cpu().numpy())
                t2.append(state_vec.cpu().numpy())

        return np.vstack(t1), np.vstack(t2)
    def predict(self, np_data, truth_label =None):
        pre_data = self.preprocess(np_data)
        data = torch.from_numpy(pre_data)

        # Test the model
        with torch.no_grad():
            images = data.reshape(-1, self.time_steps, self.n_inputs).to(device)
            state_vec, _ = self.model(images)
            dense = self.model.fc(state_vec)
            # print(dense.shape)
            # _,
            sv_softmax = torch.nn.functional.softmax(dense, dim=-1)
        sv_softmax= sv_softmax.cpu().numpy()
        state_vec = state_vec.cpu().numpy()
        return np.argmax(sv_softmax, axis=-1)[:, -1], np.argmax(sv_softmax, axis=-1), sv_softmax, state_vec, truth_label


if __name__ == "__main__":
    classifier = TorchMnistiClassifier(rnn_type='lstm', save_dir='../../data/mnist_rnn_profile_torch', flip=1,train_default=True, epoch=20)
    # arr = np.random.choice(60000, 500, replace=False)
    # classifier.train(removed_idx=arr)
    # classifier.train()
    # from tensorflow import keras
    #
    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # test = x_test[0:1]
    # y_te = y_test[0:1]
    # l,p = classifier.cal_loss(test,y_te,True)
    # print('')
    # print(os.path.dirname('a/b/c/a'))
    # classifier = TorchMnistiClassifier(rnn_type='lstm', save_dir='../../data/mnist_rnn_torch')
    #
    # test_dataset = torchvision.datasets.MNIST(root=mnist_dir,
    #                                           train=False,

    #                                           transform=transforms.ToTensor())
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                           batch_size=100,
    #                                           shuffle=False)
    # y = []
    # x = []
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #
    #     for images, labels in test_loader:
    #         images = images.reshape(-1, classifier.time_steps, classifier.n_inputs).to(device)
    #         labels = labels.to(device)
    #         _, outputs = classifier.model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         y.append(predicted.numpy())
    #         x.append(labels.numpy())
    #
    #
    #
    #     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    # from tensorflow.keras.datasets import mnist
    # from torchsummary import summary
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # #
    # # # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # outputs = classifier.predict_np(x_test)
    # res = outputs[0][:, -1]
    # #
    # # x0 = np.concatenate(x)
    # # y0 = np.concatenate(y)
    # same = np.sum(res == y_test)
    # # correct = np.sum(x == y0)
    # #
    #
    # print(same)
