
import sys
import joblib
import numpy as np
import os
import argparse
import torch

from RNNRepair.utils import create_args,get_project_path
from RNNRepair.utils import calculate_similarity_list, calculate_jcard_list

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

    return total_seq
if __name__ == "__main__":
    parser = create_args()
    parser.add_argument('-benign', default=0, type=int)
    args = parser.parse_args()
    
    if args.model == 'keras_lstm_mnist' or args.model == 'keras_gru_mnist':
        class_num = 10
    else:
        class_num =2
    # save_dir = os.path.join(get_project_root(), 'data', args.model)
    save_dir = get_project_path(args.path,args.model) 
    
    abst_dir = os.path.join(save_dir, 'abs_model')
    model_path = os.path.join(abst_dir, str(args.pca) + '_' + str(args.epoch) + '_' + str(args.components) + '_GMM.ast')
    gan_train = os.path.join(save_dir, 'feature_data',
                             str(args.pca) + '_' + str(args.epoch) + '_' + str(args.components) + '_feature.npz')

    train_trace, test_trace = joblib.load(gan_train)

    pca_dir = os.path.join(save_dir, 'pca_trace')

    k_path = os.path.join(pca_dir, str(args.pca) + '_' + str(args.epoch) + '.ptr')
    pca_data, softmax, pred_seq_labels, train_pred_labels, train_truth_labels = joblib.load(k_path)
    k_path = os.path.join(pca_dir, str(args.pca) + '_' + str(args.epoch) + '.tptr')
    test_pca_data, test_softmax, test_seq_labels, test_pred_labels, test_truth_labels = joblib.load(k_path)
    if args.benign == 0:
        target_test_indexes = np.where(test_pred_labels != test_truth_labels)[0]
    else:
        target_test_indexes = np.where(test_pred_labels == test_truth_labels)[0]
        target_test_indexes = np.random.choice(target_test_indexes, 200, replace=False)
    test_pred_results = test_pred_labels[target_test_indexes]
    print('Found ', len(target_test_indexes))


    top_n = 100
    best_model = joblib.load(model_path)
    name = '_faults.log' if args.benign == 0 else '_benign.log'
    log_path = os.path.join(abst_dir, str(args.pca) + '_' + str(args.epoch) + name)
    plot_file = open(log_path, 'a+')

    total_pred = [[] for i in range(top_n)]
    total_truth = [[] for i in range(top_n)]
    total_pred_sim = [[] for i in range(top_n)]
    total_truth_sim = [[] for i in range(top_n)]
    min_pred = [[] for i in range(top_n)]
    min_truth = [[] for i in range(top_n)]
    with torch.no_grad():
        total_results = []
        for i, index in enumerate(target_test_indexes):

            pred_label = test_pred_results[i]
            truth_label = test_truth_labels[index]

            # i_tr = np.expand_dims(test_trace[index], axis=0)
            # print(train_trace[0].shape)
            # i_traces = np.concatenate([i_tr] * len(train_trace), axis=0)

            similarities = calculate_similarity_list(test_trace[index], train_trace, best_model.m, class_num)
            sort_indexes = np.argsort(similarities)
            close_labels = train_truth_labels[sort_indexes]


            for j in range(top_n):

                truth_indexes = np.where(close_labels[0:(j+1)] == truth_label)[0]
                pred_indexes = np.where(close_labels[0:(j+1)] == pred_label)[0]


                pred_in_train = sort_indexes[pred_indexes] #indexes == pred label
                truth_in_train = sort_indexes[truth_indexes] #indexes == train label

                truth_sims = similarities[sort_indexes[truth_indexes]]
                pred_sims = similarities[sort_indexes[pred_indexes]]


                total_pred[j].append(len(pred_in_train))
                total_truth[j].append(len(truth_in_train))

                if len(pred_in_train) > 0:
                    total_pred_sim[j].append(np.average(pred_sims))
                    min_pred[j].append(pred_sims[0])
                if len(truth_in_train) > 0:
                    total_truth_sim[j].append(np.average(truth_sims))
                    min_truth[j].append(truth_sims[0])



        for j in range(top_n):
            if args.benign == 1:
                plot_file.write(
                    "%f,%f,%f\n" %
                    (float(len(total_truth_sim[j]))/len(target_test_indexes),
                     np.average(total_truth[j]),
                     np.average(total_truth_sim[j])
                     ))
            else:
                plot_file.write(
                    "%f,%f,%f,%f,%f,%f\n" %
                    (float(len(total_truth_sim[j]))/len(target_test_indexes),
                     np.average(total_truth[j]),
                     np.average(total_truth_sim[j]),
                     float(len(total_pred_sim[j]))/len(target_test_indexes),
                     np.average(total_pred[j]),
                     np.average(total_pred_sim[j])
                     ))
            plot_file.flush()

        plot_file.close()
        print('finish')