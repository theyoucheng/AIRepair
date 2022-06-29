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

from ...profile_abs import  Profiling
from .mutators import Mutators

class MnistClassifier(Profiling):
    def __init__(self, rnn_type, save_dir, epoch = 5, save_period = 0, overwrite = False):
        # Classifier
        super().__init__(save_dir)
        self.time_steps = 28  # timesteps to unroll
        self.n_units = 128  # hidden LSTM units
        self.n_inputs = 28  # rows of 28 pixels (an mnist img is 28x28)
        self.n_classes = 10  # mnist classes/labels (0-9)
        self.batch_size = 128  # Size of each batch
        self.channel = 1
        self.n_epochs = epoch
        # Internal

        self.rnn_type = rnn_type
        self._period = save_period

        self.model_path = os.path.join(self.model_dir, (rnn_type+'_%d.h5')%(epoch))
        if (not overwrite) and os.path.exists(self.model_path):
            print('loaded existing model', self.model_path)
            self.model = self.create_model_hidden(os.path.join(self.model_path))
        else:
            self.train()

    def create_model(self):
        model = Sequential()
        if self.rnn_type == 'lstm':
            model.add(LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
        elif self.rnn_type == 'gru':
            model.add(GRU(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        return model

    # create model which outputs hidden states
    def create_model_hidden(self, path):

        input = Input(shape=(self.time_steps, self.n_inputs))
        if self.rnn_type == 'lstm':
            rnn = LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs), return_sequences=True)(input)
        elif self.rnn_type == 'gru':
            rnn = GRU(self.n_units, input_shape=(self.time_steps, self.n_inputs), return_sequences=True)(input)
        else:
            assert False
        each_timestep = rnn
        dense = Dense(10, activation='softmax')(each_timestep)
        model = Model(inputs=input, outputs=[dense, rnn])
        model.load_weights(path)
        return model



    def train(self, new_data=None, new_label=None):
        self.model = self.create_model()

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if new_data is not None:
            x_train = np.append(x_train, new_data, axis=0)
            y_train = np.append(y_train, new_label, axis=0)


        x_train = x_train.reshape(x_train.shape[0], self.n_inputs, self.n_inputs)
        x_test = x_test.reshape(x_test.shape[0], self.n_inputs, self.n_inputs)

        y_test= keras.utils.to_categorical(y_test, num_classes=10)
        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        filepath = self.model_dir + '/' + self.rnn_type + "_{epoch:d}.h5"

        calls = []
        if self._period > 0:
            mc = keras.callbacks.ModelCheckpoint(filepath,
                                             save_weights_only=True, period=self._period)
            calls.append(mc)

        self.model.fit(x_train, y_train, validation_data=(x_test, y_test),
                       batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False, callbacks=calls,verbose=2)
        model  = self.model
        if self._period == 0:
            self.model.save(self.model_path)
        self.model = self.create_model_hidden(self.model_path)
        return model
    def do_profile(self, test=False):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if test:
            return self.predict(x_test, y_test)
        else:
            return self.predict(x_train, y_train)

    def preprocess(self, x_test):
        x_test = x_test.astype('float32')
        x_test /= 255
        return x_test


    def predict(self, data, truth_label=None):
        if data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)
        data = self.preprocess(data)
        sv_softmax, state_vec = self.model.predict(data)
        return np.argmax(sv_softmax, axis=-1)[:, -1], np.argmax(sv_softmax, axis=-1), sv_softmax, state_vec, truth_label

