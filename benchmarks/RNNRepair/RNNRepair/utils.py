import os, numpy as np
import matplotlib.pyplot as plt
# from pathlib import Path
import argparse
import joblib

import tensorflow as tf 

import torch.nn.functional as F
import torch

import torch.utils.data
device =torch.device("cuda" if torch.cuda.is_available() else "cpu" )

# from . import reproducible


def get_traces(imgs, classifier, pca, best_model):
    final_labels, seq_labels, softmax, con_tr, truth_labels  = classifier.predict(imgs)

    # gen_con = gen_results[1]
    # softmax = gen_results[0]

    # gen_lbls = np.argmax(softmax, axis=-1)

    gen_pca_data = pca.do_reduction(con_tr)

    traces = best_model.get_trace_input(gen_pca_data, seq_labels)
    return traces, softmax



def pad_torch_from_numpy(np_list):
    be2torch =lambda x:torch.from_numpy(x)
    torch_list = [be2torch(x) for x in np_list]
    torch_list = torch.nn.utils.rnn.pad_sequence( torch_list,batch_first=True )
    return torch_list

def calculate_jcard_list(tr1, tr2_list):

    def get_trans(tr):
        trs = []
        comps = tr[:,0]
        pre = -1
        for i in comps:
            trs.append((pre, i))
            pre = i
        return trs
    tr1_trans = get_trans(tr1)
    sims = []
    for temp_tr in tr2_list:
        tm_tr = get_trans(temp_tr)
        common = [val for val in tr1_trans if val in tm_tr]
        sims.append(len(common)/len(tr1_trans))
    return np.array(sims)

def calculate_similarity_list(tr1, tr2_list, components, label=10):


    tr1_tensor = torch.from_numpy(tr1)

    traces = [torch.from_numpy(i) for i in tr2_list]
    traces.append(tr1_tensor)


    total = torch.nn.utils.rnn.pad_sequence(traces, batch_first=True)

    total[:,:,1] = (total[:,:,1]+1) / (label+1)
    total [:,:,0] = (total[:,:,0]+1) /(components+1)

    tr1 = total[-1]
    tr2 = total[:-1]

    tr1 = torch.unsqueeze(tr1, dim=0)
    tr1 =  torch.cat([tr1]*len(tr2), dim=0)


    # print(tr1.shape, tr2.shape)

    with torch.no_grad():
        return F.l1_loss(tr1, tr2, reduction="none").mean([-1, -2]).numpy()

def get_project_root(path=None):
    """Returns project root folder."""
    # return Path(__file__).parent
    if path is None:
        return os.path.join(os.path.dirname(__file__),"..","save")
    return path
def get_project_dataroot(path=None):
    """Returns project root folder."""
    # return Path(__file__).parent
    if path is None:
        return os.path.join(os.path.dirname(__file__),"..","data")
    return path

def get_project_path(path=None,model_type=None):
    """Returns project root folder."""
    if model_type is None :
        raise Exception("unknown model_type,",type(model_type))
    
    save_dir = get_project_root(path=path)
    return os.path.join(save_dir,model_type)

toxic_data_path = os.path.join(get_project_dataroot(),  'toxic_data','data_list')
sst_data_path = os.path.join(get_project_dataroot(),  'sst_data','data_list')
def save_image(save_dir,data_label,name):
    os.makedirs(save_dir, exist_ok=True)

    c = int(np.sqrt(data_label.shape[0]))
    r = int(np.ceil(data_label.shape[0] / c))
    fig, axs = plt.subplots(r, c)

    cnt = 0
    for i in range(r):
        for j in range(c):
            if cnt == data_label.shape[0]:
                break
            # vis = np.concatenate([gen_imgs[cnt,:,:] , data_label[cnt,:,:]] ,axis=-1)
            axs[i,j].imshow( data_label[cnt], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1

    fig.savefig("%s/%s.png" %(save_dir, name) )
    plt.close()



def save_image_cr(save_dir,data_label,name, row, col):
    os.makedirs(save_dir, exist_ok=True)
    if row == 1:
        row += 1
    fig, axs = plt.subplots(row, col)

    for i, row_i in enumerate(data_label):
        for j, img in enumerate(row_i):
            if img is None:
                continue
            axs[i, j].imshow(img, cmap='gray')
            axs[i, j].axis('off')

    fig.savefig("%s/%s.pdf" %(save_dir, name) )
    plt.close()

def save_text_cr(save_dir, data, name):
    os.makedirs(save_dir, exist_ok=True)
    s = ''
    for ls in data:
        s += '#########################\r\n'
        for tx in ls:
            s += ' '.join(tx[1])
            s += '\r\n===================\r\n'
    path = os.path.join(save_dir, name)
    plot_file = open(path, 'a+', encoding="utf-8")
    plot_file.write(s)
    plot_file.flush()
    plot_file.close()




def create_args():
    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')
    parser.add_argument('-pca', default=10, type=int)
    parser.add_argument('-epoch', default=30, type=int)
    parser.add_argument('-start', default=20, type=int)
    parser.add_argument('-end', default=60, type=int)
    parser.add_argument('-components', default=7, type=int)
    parser.add_argument('-path',default=None,)

    parser.add_argument('-model', default='torch_lstm_mnist',
                        choices=['keras_lstm_mnist', 'keras_gru_mnist','torch_lstm_mnist','torch_gru_mnist',
                                 'torch_lstm_imdb','torch_gru_imdb', 
                                 'torch_lstm_toxic', 'torch_gru_toxic',
                                 'torch_lstm_sst', 'torch_gru_sst',
                                 ])
    # args = parser.parse_args()
    #
    # save_dir = os.path.join(get_project_root() if args.path is None else args.path, 'data', args.model)
    #
    # raise Exception("model load")
    # if args.model == 'keras_lstm_mnist':
        # from use_cases.image_classification.mnist_rnn_profile import MnistClassifier
        # classifier = MnistClassifier(rnn_type='lstm', save_dir=save_dir, epoch=args.epoch)
    # elif args.model == 'keras_gru_mnist':
        # from use_cases.image_classification.mnist_rnn_profile import MnistClassifier
        # classifier = MnistClassifier(rnn_type='gru', save_dir=save_dir, epoch=args.epoch)
    # elif args.model == 'torch_lstm_mnist':
        # from use_cases.image_classification.mnist_rnn_torch_profile import TorchMnistiClassifier
        # classifier = TorchMnistiClassifier(rnn_type='lstm', save_dir=save_dir, epoch=args.epoch)
    # elif args.model == 'torch_gru_toxic':
        # from use_cases.image_classification.mnist_rnn_torch_profile import TorchMnistiClassifier
        # classifier = TorchMnistiClassifier(rnn_type='gru', save_dir=save_dir, epoch=args.epoch)
    # else:
        # assert (False)
    return parser # args, classifier

#
#
# def output_process(dense_list, state_vec_softmax_list, state_vec_list, truth_labels):
#     state_vec_softmax = []
#     seq_labels = []
#     final_labels = []
#     for i, batch in enumerate(state_vec_softmax_list):
#         state_vec_softmax.extend(list(batch))
#         seq_labels.extend(list(np.argmax(batch, axis=-1)))
#         final_labels.append(dense_list[i])
#     return np.concatenate(final_labels, axis=0), seq_labels, state_vec_softmax, state_vec_list, truth_labels