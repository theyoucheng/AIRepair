
import argparse
# Imports
from tensorflow import keras
import os
import sys
import numpy as np
import random

# sys.path.append("../../")
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,LSTM,GRU, Dense
from tensorflow.keras.models import Model

from RNNRepair.use_cases.image_classification.mutators import Mutators

from RNNRepair.use_cases.image_classification.mnist_rnn_profile_keras import MnistClassifier

if __name__ == "__main__":
    print('Repair by retraining')

    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')
    parser.add_argument('-epoch', default=15, type=int)

    # 0 for train with new data, 1 for test on all original models
    parser.add_argument('-type', default=0, choices=[0, 1, 2], type=int)

    parser.add_argument('-p', default='../../app/RQ4/retrain.npz')

    parser.add_argument('-start', default=5, type=int)
    parser.add_argument('-seed', default=1, type=int)
    parser.add_argument('-rnn_type', default='lstm')

    args = parser.parse_args()
    data = np.load(args.p)
    new_data = data['data']
    new_label = data['label']
    new_id = data['testid']

    print(new_data.shape, new_label.shape, new_id.shape)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    correct = np.zeros(len(new_id))

    train_correct = np.zeros(len(new_id))

    test_label = y_test[new_id]

    if args.rnn_type == 'lstm':
        cur_classifier = MnistClassifier(rnn_type='lstm', save_dir='./save/keras_lstm_mnist', epoch=15,save_period=0)
    else:
        cur_classifier = MnistClassifier(rnn_type='gru', save_dir='./save/keras_gru_mnist', epoch=15,save_period=0)

    seed = args.seed

    start = args.start


    aa = []
    bb = []
    cc = []
    dd = []
    (x_train1, y_train1), (x_test, y_test) = mnist.load_data()



    for j in range(seed):
        current_pred_list = []
        current_correct_list = []
        train_correct_list = []
        ori_train_list = []
        if args.type == 0:
            for i in range(start, args.epoch):
                i += 1
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                cur_classifier.n_epochs = i
                model = cur_classifier.train(new_data=new_data, new_label=new_label)

                ori_train = np.argmax(model.predict(cur_classifier.preprocess(x_train1)), axis=-1)
                ori_train_list.append(np.sum(ori_train == y_train1))

                pl = np.argmax(model.predict(cur_classifier.preprocess(x_test[new_id])), axis=-1)
                correct += (pl == test_label)

                train_data = cur_classifier.preprocess(new_data)
                train_pl = np.argmax(model.predict(train_data), axis=-1)

                print('Truth  label', test_label)
                print('Current pred', pl)
                print('Correct num ', np.sum(test_label == pl))
                # print('Train  pred', train_pl)
                print('Train Correct', np.sum(train_pl == new_label))

                current_pred_list.append(pl == test_label)
                current_correct_list.append(np.sum(test_label == pl))
                train_correct_list.append(np.sum(train_pl == new_label))

                del model
            print('Final', correct)
        elif args.type == 2:

            rand_data = []
            imgs = np.expand_dims(new_data, axis=-1)
            rotate = list(range(-100, 10))
            rotate.extend(list(range(10,100)))
            for img in imgs:
                i = random.choice(rotate)
                rand_data.append(Mutators.image_rotation(img, i))

            for i in range(start, args.epoch):
                i += 1
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                cur_classifier.n_epochs = i
                model = cur_classifier.train(new_data=rand_data, new_label=new_label)

                ori_train = np.argmax(model.predict(cur_classifier.preprocess(x_train1)), axis=-1)
                ori_train_list.append(np.sum(ori_train == y_train1))

                pl = np.argmax(model.predict(cur_classifier.preprocess(x_test[new_id])), axis=-1)
                correct += (pl == test_label)

                train_data = cur_classifier.preprocess(new_data)
                train_pl = np.argmax(model.predict(train_data), axis=-1)

                print('Truth  label', test_label)
                print('Current pred', pl)
                print('Correct num ', np.sum(test_label == pl))
                # print('Train  pred', train_pl)
                print('Train Correct', np.sum(train_pl == new_label))

                current_pred_list.append(pl == test_label)
                current_correct_list.append(np.sum(test_label == pl))
                train_correct_list.append(np.sum(train_pl == new_label))

                del model

            print('Final', correct)
        elif args.type == 1:

            for i in range(start, args.epoch):
                i += 1
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                cur_classifier.n_epochs = i
                model = cur_classifier.train()

                pl = np.argmax(model.predict(cur_classifier.preprocess(x_test[new_id])), axis=-1)
                correct += (pl == test_label)

                train_data = cur_classifier.preprocess(new_data)
                train_pl = np.argmax(model.predict(train_data), axis=-1)

                print('Truth  label', test_label)
                print('Current pred', pl)
                print('Correct num ', np.sum(test_label == pl))
                # print('Train  pred', train_pl)
                print('Train Correct', np.sum(train_pl == new_label))

                current_pred_list.append(pl == test_label)
                current_correct_list.append(np.sum(test_label == pl))
                train_correct_list.append(np.sum(train_pl == new_label))

                ori_train = np.argmax(model.predict(cur_classifier.preprocess(x_train1)), axis=-1)
                ori_train_list.append(np.sum(ori_train == y_train1))
                del model
            print('Final', correct)

        current_pred = np.array(current_pred_list)
        q = np.sum(current_pred, axis=0)
        current_pred_avg = np.sum(current_pred, axis=0) / (args.epoch - start)
        current_correct_avg = sum(current_correct_list) / (args.epoch - start)
        train_correct_avg = sum(train_correct_list) / (args.epoch - start)

        ori_train_avg = sum(ori_train_list) / (args.epoch - start)

        aa.append(current_pred_avg)
        bb.append(current_correct_avg)
        cc.append(train_correct_avg)
        dd.append(ori_train_avg)

    aa = np.array(aa)
    bb = np.array(bb)
    cc = np.array(cc)
    dd = np.array(dd)
    print('\r\n\r\n\r\n')
    # print('===========pred in each seed===========')
    # print(aa)
    # print('===========correct in each seed===========')
    # print(bb)
    # print('===========train in each seed===========')
    # print(cc)

    print('===========All avg===========')
    print(np.sum(aa, axis=0) / seed)

    print('Correct Avg', np.sum(bb) / seed)
    # print(np.sum(cc) / seed)
    # print(np.sum(dd) / (seed * len(x_train1)))
