import sys
# sys.path.append("../../")

import joblib
import numpy as np

import os
import argparse
import torch
import shutil
from collections import Counter
from keras.datasets import mnist
import time

from RNNRepair.utils import calculate_similarity_list
from RNNRepair.abstraction.feature_extraction import extract_feature
from RNNRepair.use_cases.image_classification.mnist_rnn_profile_torch import TorchMnistiClassifier


def inference_key(trace, trans_train_without, pred_results, pred, truth):
    abst_states = trace[:, 0].astype(int)  # self.gmm.predict(trace)

    pred_labels = trace[:, 1].astype(int)
    total_len = len(abst_states)
    pred_labels = np.insert(pred_labels, 0, 0)
    abst_states = np.insert(abst_states, 0, 0)

    sequence_num = []
    total_seq = []
    pred_ratio = []
    truth_ratio = []

    n = total_len - 1
    err = pred_labels[total_len]
    while n > 0:
        if pred_labels[n] != err:
            break
        n -= 1


    key =  (abst_states[n], pred_labels[n], pred_labels[n+1], abst_states[n+1])
    train = list(trans_train_without[key])
    return train
    
def inference(trace, trans_train_without, pred_results, pred, truth):
    abst_states = trace[:, 0].astype(int)  # self.gmm.predict(trace)

    pred_labels = trace[:, 1].astype(int)
    total_len = len(abst_states)
    pred_labels = np.insert(pred_labels, 0, 0)
    abst_states = np.insert(abst_states, 0, 0)

    sequence_num = []
    total_seq = []
    pred_ratio = []
    truth_ratio = []

    for i, lb in enumerate(abst_states):
        if i + 1 < total_len:
            key = (lb, pred_labels[i], pred_labels[i+1], abst_states[i+1])
            if key in trans_train_without:
                train  = list(trans_train_without[key])
                # print(train)
                sequence_num.append(len(train))
                pred_ratio.append(np.sum(pred_results[train] == pred))
                truth_ratio.append(np.sum(pred_results[train] == truth))
                total_seq.extend(train)
            else:
                sequence_num.append(0)
                pred_ratio.append(0)
                truth_ratio.append(0)

    return total_seq#, np.array(sequence_num), np.array(pred_ratio), np.array(truth_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')
    parser.add_argument('-pca', default=10, type=int)
    parser.add_argument('-epoch', default=20, type=int)
    parser.add_argument('-components', default=32, type=int)
    parser.add_argument('-flip', default=1, type = int)
    parser.add_argument('-flipfirst', default=1, type=int)
    parser.add_argument('-flipsecond', default=7, type=int)
    parser.add_argument('-start_seed', default=1000, type=int)
    parser.add_argument('-end_seed', default=1010, type=int)
    parser.add_argument('-path')
    parser.add_argument('-ratio', default=0.3, type=float)
    args = parser.parse_args()
    pca_dim = args.pca
    comp = args.components
    epoch = args.epoch
    
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    
    log_file = open("{}/log.file".format(args.path),"w+")
    origin_time = time.time()
    seeds = np.arange(args.start_seed,args.end_seed)
    for cur_seed in seeds:
        print("\n***********cur_seed:{}***********".format(cur_seed))
        print("\n***********cur_seed:{}***********".format(cur_seed),file=log_file)
        no_flip_save_dir = os.path.join(args.path, 'torch_lstm_bin', str(cur_seed)+'_0')
        classifier0 = TorchMnistiClassifier(rnn_type='lstm', save_dir=no_flip_save_dir, epoch=args.epoch, seed=cur_seed, first=args.flipfirst,
                                            second=args.flipsecond, ratio = args.ratio)
        p,_,_,_,t = classifier0.do_profile(test=True)

        flip_save_dir = os.path.join(args.path, 'torch_lstm_bin', str(cur_seed)+'_'+str(args.flip))
        classifier = TorchMnistiClassifier(rnn_type='lstm', save_dir=flip_save_dir, epoch=args.epoch, seed=cur_seed, flip=args.flip, first=args.flipfirst,
                                            second=args.flipsecond, ratio = args.ratio)

        (_,_), (_, y1), _ = classifier0.load_binary_data()
        (_,_), (x2, y2), flip_idx = classifier.load_binary_data()
        assert(np.sum(y1!=y2) == 0)
        assert(np.sum(t!=np.argmax(y2, axis=-1)) == 0)
        # exit(0)
        # assert False,"generate flip data"
        start_time = time.time()
        pca, ast_model = classifier.build_abstract_model(pca_dim, comp, epoch, 'GMM', classifier.n_classes)
        pca, pca_data, softmax, pred_seq_labels, train_pred_labels, train_truth_labels = classifier.get_pca_traces(
            pca_dim, epoch)
        test_pca_data, test_softmax, test_seq_labels, test_pred_labels, test_truth_labels = classifier.get_test_pca_traces(
            pca, pca_dim, epoch)
        train_trace, test_trace = extract_feature(ast_model, classifier, pca_dim, epoch, pca_data, pred_seq_labels,
                                                test_pca_data, test_seq_labels, gmm=comp)
        transition_path = os.path.join(classifier.abst_dir,
                                    str(pca_dim) + '_' + str(epoch) + '_' + str(comp) + '_trans.data')
        if os.path.exists(transition_path):
            transition, transition_train, transition_train_without = joblib.load(transition_path)
        else:
            print('Start build transitions')
            transition, transition_train, transition_train_without = ast_model.update_transitions(train_trace,
                                                                                                classifier.n_classes,
                                                                                                None)
            joblib.dump((transition, transition_train, transition_train_without), transition_path)
            print('End build transitions')
        print("[num]flip_idx:",len(flip_idx))
        print("[num]flip_idx:",len(flip_idx),file=log_file)
        real_ = np.where(train_pred_labels[flip_idx] == train_truth_labels[flip_idx])[0]
        flip_idx = flip_idx[real_]
        print("[num]infl_flip_idx:",len(flip_idx))
        print("[num]infl_flip_idx:",len(flip_idx),file=log_file)
        save_fn = "{}/infl_flip_idx({}_{}_{}_{}_{}).npy".format(args.path,args.flipfirst,args.flipsecond,args.flip,args.ratio,cur_seed)
        if not os.path.exists(save_fn):
            np.save(save_fn,flip_idx)
            print("[save]Save {} successfully!".format(save_fn))
            print("[save]Save {} successfully!".format(save_fn),file=log_file)
        print("[build]Time:{:.4f}".format(time.time()-start_time))
        print("[build]Time:{:.4f}".format(time.time()-start_time),file=log_file)
        
        if args.flip == 1: #0 -> 1
            cond0 = test_pred_labels == 1
            cond = np.logical_and(cond0, np.logical_and(p == t, test_pred_labels != test_truth_labels))
        elif args.flip == 2:
            cond0 = test_pred_labels == 0
            cond = np.logical_and(cond0, np.logical_and(p == t, test_pred_labels != test_truth_labels))
        else:
            cond = np.logical_and(p == t, test_pred_labels != test_truth_labels)


        target_test_indexes = np.where(cond)[0]
        t_preds, _, _, _, t_truth_label = classifier.predict(x2[target_test_indexes],y2[target_test_indexes])# <<<<<<<<<<<<<<<<<<
        t_truth_label = np.argmax(t_truth_label,axis=1)
        assert np.sum(np.logical_and(t_truth_label,t_preds))==0
        print("[num]target_test:{}".format(len(target_test_indexes)))
        print("[num]target_test:{}".format(len(target_test_indexes)),file=log_file)
        save_fn = "{}/target_test_{}_{}_{}_{}_{}.npy".format(args.path,args.flipfirst,args.flipsecond,args.flip,args.ratio,cur_seed)
        if not os.path.exists(save_fn):
            np.save(save_fn,target_test_indexes)
            print("[save]Save {} successfully!".format(save_fn))
            print("[save]Save {} successfully!".format(save_fn),file=log_file)

        start_time = time.time()
        tt = 27
        t_order = np.zeros(len(train_truth_labels), dtype=float)
        s_order = np.zeros(len(train_truth_labels), dtype=float)
        for i, index in enumerate(target_test_indexes):
            pred_label = int(test_pred_labels[index])
            truth_label = int(test_truth_labels[index])
            total_seq = inference(test_trace[index], transition_train_without,
                                                train_pred_labels, pred_label, truth_label)

            train_items = Counter(total_seq)

            most_com = train_items.most_common()
            for (m, n) in most_com:
                # train_order.append(m)
                t_order[m] += n/tt
            similarities = calculate_similarity_list(test_trace[index], train_trace, ast_model.m, classifier.n_classes)
            s_order += similarities

        print("[calc]Time:{:.4f}".format(time.time()-start_time))
        print("[calc]Time:{:.4f}".format(time.time()-start_time),file=log_file)

        sort_indexes =  np.argsort(similarities)
        train_indexes = np.argsort(t_order*-1)
        save_fn = "{}/sim_order_{}_{}_{}_{}_{}.npy".format(args.path,args.flipfirst,args.flipsecond,args.flip,args.ratio,cur_seed)
        save_fn_2 = "{}/sim_scores_{}_{}_{}_{}_{}.npy".format(args.path,args.flipfirst,args.flipsecond,args.flip,args.ratio,cur_seed)
        if not os.path.exists(save_fn_2):
            np.save(save_fn,sort_indexes)
            np.save(save_fn_2,similarities)
            print("[save]Save {} successfully!".format(save_fn))
            print("[save]Save {} successfully!".format(save_fn_2))
        save_fn = "{}/jcard_order_{}_{}_{}_{}_{}.npy".format(args.path,args.flipfirst,args.flipsecond,args.flip,args.ratio,cur_seed)
        save_fn_2 = "{}/jcard_scores_{}_{}_{}_{}_{}.npy".format(args.path,args.flipfirst,args.flipsecond,args.flip,args.ratio,cur_seed)
        if not os.path.exists(save_fn_2):
            np.save(save_fn,train_indexes)
            np.save(save_fn_2,t_order*-1)
            print("[save]Save {} successfully!".format(save_fn))
            print("[save]Save {} successfully!".format(save_fn_2))
    

        rr= (np.arange(20) + 1) * 0.05
        flip_inf = []
        flip_sim = []
        total_train = len(sort_indexes)
        len_ind = []
        for j in rr:
            infl_indexes = train_indexes[0:int(total_train * j)]
            sim_indexes = sort_indexes[0:int(total_train * j)]

            flip_inf.append(len(np.intersect1d(flip_idx, infl_indexes))/len(flip_idx))
            flip_sim.append(len(np.intersect1d(flip_idx, sim_indexes))/ len(flip_idx))
            len_ind.append(int(total_train * j))

        print(">>>>>>>>>>>>>>>>>>>>")
        print(np.around(flip_inf,4))
        print(np.around(flip_sim,4))
        print(len(flip_idx))
        print("<<<<<<<<<<<<<<<<<<<")

        print(">>>>>>>>>>>>>>>>>>>>",file=log_file)
        print(np.around(flip_inf,4),file=log_file)
        print(np.around(flip_sim,4),file=log_file)
        print(len(flip_idx),file=log_file)
        print("<<<<<<<<<<<<<<<<<<<",file=log_file)
    
    print("Total time for seeds {} : {:.4f}\n".format(seeds,time.time()-origin_time))
    print("Total time for seeds {} : {:.4f}\n".format(seeds,time.time()-origin_time),file=log_file)

    log_file.close()






