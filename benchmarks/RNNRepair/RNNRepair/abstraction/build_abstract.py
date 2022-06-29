import os
import numpy as np
from ..utils import get_project_root
from ..use_cases import create_classifer
class AbstConstructor():
    def __init__(self, pca, epoch, components, path, model_type, dataset=None):
        self.pca_num = pca
        self.epoch = epoch
        self.components = components
        self.path = path
        self.rnn_type = model_type
        self.dataset = dataset

        # self.save_dir = os.path.join(get_project_root() if self.path is None else self.path, 'data', self.rnn_type)
        self.save_dir = os.path.join(get_project_root() if self.path is None else self.path,  self.rnn_type)

        self.classifier = create_classifer(
            self.rnn_type,
            save_dir=get_project_root(), epoch=self.epoch)
        # if self.rnn_type == 'keras_lstm_mnist':
            # from use_cases.image_classification.mnist_rnn_profile import MnistClassifier
            # self.classifier = MnistClassifier(rnn_type='lstm', save_dir=self.save_dir, epoch=self.epoch)
        # elif self.rnn_type == 'torch_lstm_imdb':
            # from use_cases.sentiment_analysis.imdb_rnn_profile import IMDBClassifier
            # self.classifier = IMDBClassifier(rnn_type='lstm', save_dir=self.save_dir, epoch=self.epoch, train_default = False)
        # elif self.rnn_type == 'torch_gru_toxic':
            # from use_cases.sentiment_analysis.toxic_rnn_profile import TOXICClassifier
            # self.classifier = TOXICClassifier(rnn_type='gru', save_dir=self.save_dir, epoch=self.epoch, train_default = False)
        # elif self.rnn_type == 'torch_gru_sst':
            # from use_cases.sentiment_analysis.sst_rnn_profile import SSTClassifier
            # self.classifier = SSTClassifier(rnn_type='gru', save_dir=self.save_dir, epoch=self.epoch, train_default = False)
        # elif self.rnn_type == 'torch_lstm_bin':
            # from use_cases.image_classification.mnist_rnn_profile_torch import TorchMnistiClassifier
            #
            # self.classifier = TorchMnistiClassifier(rnn_type='lstm', save_dir=self.save_dir, epoch=self.epoch,
                                               # flip=0, first=4, second=9, ratio=0.3)
        # else:
            # assert (False)

        self.classifier.train(dataset=self.dataset, is_load=True, saved_path=self.classifier.model_path)
        self.pca, self.ast_model = self.classifier.build_abstract_model(self.pca_num, components, self.epoch, 'GMM', self.classifier.n_classes)
    def visualize_trace(self, x_test, path=None):
        x_pca_data, softmax, seq_labels, pred_labels, truth_labels, texts, texts_emd = self.classifier.cal_pca_trace(self.pca, x_test)
        t_softmax = [np.max(soft, axis=-1) for soft in softmax]
        x_test_trace = self.ast_model.get_trace_input(x_pca_data, seq_labels)

        for i in range(len(x_test)):
            pred_label = pred_labels[i]
            truth_label = truth_labels[i]
            traces, probs, names = [x_test_trace[i]], [t_softmax[i]], ['Self (blue)']
            word, word_emd = [texts[i][1]], [texts_emd[i]]
            g = self.ast_model.visualize_inputs(
                os.path.join(str(path) +'/trace_' + str(i) + '_' + str(pred_label) + '_' + str(truth_label) + '.gv'),
                traces, probs, names, word, word_emd, None)
            g.render(format='png')



