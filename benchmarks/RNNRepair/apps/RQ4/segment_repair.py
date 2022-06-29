
import  time
import sys
# sys.path.append("../../")
import joblib
import numpy as np
import copy
import os
import argparse
import torch
import random

from RNNRepair.utils import create_args,get_project_path,get_project_root
from RNNRepair.utils import calculate_similarity_list, calculate_jcard_list
from RNNRepair.abstraction.build_abstract import AbstConstructor
from RNNRepair.abstraction.feature_extraction import extract_feature

from RNNRepair.use_cases.sentiment_analysis.toxic_rnn_profile import TOXICClassifier
from RNNRepair.use_cases.sentiment_analysis.sst_rnn_profile import SSTClassifier
from RNNRepair.use_cases.sentiment_analysis.imdb_rnn_profile import IMDBClassifier


def encode(trace, emd, start, end):
    abst_states = trace[:, 0].astype(int)  # self.gmm.predict(trace)
    pred_labels = trace[:, 1].astype(int)
    pred_labels = np.insert(pred_labels, 0, 0)
    abst_states = np.insert(abst_states, 0, 0)
    result = ''

    for i in range(start, end):
        middel = '-' if emd is None else '-' + str(emd[i]) + '-'
        result += (str(abst_states[i]) + '-' + str(pred_labels[i]) + middel + str(abst_states[i + 1]) + '-' + str(
            pred_labels[i + 1])) + '#'
    return result
def cal_most_similar(test_trace, test_emd, train_trace, train_emd, test_end):

    start = test_end - 1
    re = 0
    x = -1
    while start > -1:
        str1 = encode(test_trace, test_emd, start, test_end)
        str2 = encode(train_trace, train_emd, 0, len(train_trace))
        index = str2.find(str1)
        if index == -1:
            break
        else:
            strs = str2[:index]
            re = test_end - start
            x = strs.count('#') + re
        start -= 1
    return re, x, str1


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

def get_reachable_train(trans_train, pred_label, target_state):
    if target_state == 0:
        return set()
    result = set()
    for state, value in trans_train.items():
        (_,_,_,lb,final) = state
        if final == target_state and lb == pred_label:
            result.update(value)
    return result

def find_key (trace, trans_train, correct_label):
    abst_states = trace[:, 0].astype(int)  # self.gmm.predict(trace)

    pred_labels = trace[:, 1].astype(int)
    total_len = len(abst_states)

    abst_states = np.insert(abst_states, 0, 0)
    pred_labels = np.insert(pred_labels, 0, 0)

    j = total_len - 1
    while j > -1:
        if (pred_labels[j] == correct_label or j == 0) and pred_labels[j + 1] != correct_label:
            result =  get_reachable_train(trans_train, pred_labels[j], abst_states[j])
            return result, j, abst_states[j], pred_labels[j]
        j -= 1
    print('Not found')
    return None, j, None, None

def augment(trace, train_words, prev_state, prev_label, word, is_random=False):
    abst_states = trace[:, 0].astype(int)  # self.gmm.predict(trace)
    pred_labels = trace[:, 1].astype(int)
    abst_states = np.insert(abst_states, 0, 0)
    pred_labels = np.insert(pred_labels, 0, 0)

    total_len = len(abst_states)
    # random.seed(time.time())
    if is_random:
        insert_index = random.randint(0, total_len)
        train_words[1].insert(insert_index, word)
        return None
    else:
        for i in range(total_len):
            if abst_states[i] == prev_state and pred_labels[i] == prev_label:
                train_words[1].insert(i, word)
                return None

    return None





def get_str(item, sim, p_l, t_l, trace, trans, x_emd):

    abst_states = trace[:, 0].astype(int)  # self.gmm.predict(trace)
    pred_labels = trace[:, 1].astype(int)
    total_len = len(abst_states)


    label = item[0]
    strs = item[1]
    t = str(sim)
    t += 'GroundTruth: ' + str(t_l) + ' | Pred: ' + str(p_l) + ' | Save: ' + str(label) +' : '
    t += ' '.join(strs)
    t += '\r\n'

    t += 'Init -> '

    pre_state = 0
    pre_label = 0

    for i in range(total_len):
        # key = (pre_state, pre_label, pred_labels[i], abst_states[i])

        emd_trans_sum = trans[pre_state][pre_label][x_emd[i]][pred_labels[i]][abst_states[i]]

        pre_state = abst_states[i]
        pre_label = pred_labels[i]

        t += ('['+ strs[i]+', '+ str(x_emd[i]) +'] （' + str(abst_states[i]) + ', ' + str(pred_labels[i])) + ', '+ str(emd_trans_sum)+ '） -> '
    return t + '\r\n'





def repair(current_model, pca, epoch, components, x_test, x_test_pred, x_test_truth, x_emds, x_test_trace, x_train, train_trace, top_n, random_r=False):

    transition_path = os.path.join(current_model.classifier.abst_dir,
                                   str(pca) + '_' + str(epoch) + '_' + str(components) + '_trans.data')
    if os.path.exists(transition_path):
        transition, transition_train = joblib.load(transition_path)
    else:
        print('Start build transitions')
        transition, transition_train, _ = current_model.ast_model.update_transitions(train_trace, class_num, x_train_emd)
        joblib.dump((transition, transition_train), transition_path)
        print('End build transtions')


    str_result = ''

    target_repaired = []
    with torch.no_grad():
        for i in range(len(x_test)):
            xemd = x_emds[i]
            if np.any(xemd == 0):
                print("Fail embeding")
                continue
            pred_label = x_test_pred[i]
            truth_label = x_test_truth[i]
            x_trace = x_test_trace[i]


            str_result += (str(i) + ':' + get_str(x_test[i], 0, pred_label, truth_label,x_trace ,
                                                      transition, xemd))
            str_result += '\r\n'
            train_index, test_end, prev_state, prev_label = find_key(x_trace, transition_train, truth_label)
            if train_index is None or len(train_index) == 0:
                continue
            # assert (train_index is not None and len(train_index) > 0)
            list_train_found = np.array(list(train_index))
            real_ = np.where(train_truth_labels[list_train_found] == truth_label)[0]
            found_train = list_train_found[real_]
            if len(real_) == 0:
                print("Did not find!")
                continue
            print(truth_label, 'Found potential err', x_test[i][1][test_end],
                  'Number of train can reach the previous state:', len(real_))

            str_result += '===================train =====================\r\n'

            simlarities = []
            train_ends = []
            matched_substr = []

            for ind in found_train:
                # s, id = cal_most_similar(test_trace[index], x_test_emd[index], train_trace[ind],
                #                          x_train_emd[ind], test_end)
                s, id, txt = cal_most_similar(x_trace, xemd, train_trace[ind],
                                         x_train_emd[ind], test_end)
                simlarities.append(s)
                train_ends.append(id)
                matched_substr.append(txt)
            simlarities = np.array(simlarities)
            train_ends = np.array(train_ends)


            if random_r:
                random.seed(time.time())
                np.random.seed(int(time.time()))
                sim = np.random.choice(len(simlarities), top_n, replace=False)
                print(sim)
                print('rand', simlarities[sim])
                selected_indexes = found_train[sim]
                train_ends = train_ends[sim]
                matched_substr = [matched_substr[x] for x in sim]
            else:
                sort_sim = np.argsort(simlarities * -1)[:top_n]
                print(simlarities[sort_sim])
                selected_indexes = found_train[sort_sim]
                # simlarities = simlarities[sort_sim]
                train_ends = train_ends[sort_sim]
                matched_substr = [matched_substr[x] for x in sort_sim]

            for j, ind in enumerate(selected_indexes):
                total_len = len(train_trace[ind]) + 1
                insert_index = random.randint(0, total_len)
                insert_index = train_ends[j]
                x_train[ind][1].insert(insert_index, x_test[i][1][test_end])
                str_result += ' '.join(x_train[ind][1])
                str_result += '\r\n'
                str_result += matched_substr[j] + '\r\n'


            #
            # for ind in found_train[:top_n]:
            #     augment(train_trace[ind], x_train[ind], prev_state, prev_label, x_test[i][1][test_end], is_random=False)
            #     str_result += ' '.join(x_train[ind][1])
            #     str_result += '\r\n'
            target_repaired.append(i)

    target_test = [x_test[i] for i in target_repaired]
    new_path = os.path.join(current_model.save_dir, 'repair')
    os.path.exists(new_path) or os.makedirs(new_path)

    return str_result, target_test, x_train

    # new_model = AbstConstructor(pca, epoch, components, new_path, current_model.model_type, dataset=(x_train, target_test))
    # return new_model, str_result, target_test





if __name__ == "__main__":
    parser= create_args()
    # parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')
    # parser.add_argument('-pca', default=10, type=int)
    # parser.add_argument('-epoch', default=2, type=int)
    # parser.add_argument('-components', default=22, type=int)
    parser.add_argument('-benign', default=0, type=int)
    parser.add_argument('-rd', default=1, type=int)
    parser.add_argument('-rand', default=10, type=int)
    # parser.add_argument('-model', default='torch_lstm_imdb',
                        # choices=['keras_lstm_mnist', 'torch_lstm_imdb', 'torch_gru_toxic', 'torch_lstm_bin' , 'torch_gru_sst', ])
    # parser.add_argument('-path', default='./data')
    args = parser.parse_args()
    
    save_dir = get_project_root()

    first_abst = AbstConstructor(args.pca,args.epoch,args.components,args.path,args.model)
    if args.model == 'keras_lstm_mnist' or args.model == 'keras_gru_mnist':
        class_num = 10
        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test_emd = x_test
        x_train_emd = x_train
    else:
        class_num = 2
        texts_path = os.path.join(first_abst.save_dir, '_texts')
        x_test = joblib.load(os.path.join(texts_path, 'test.texts'))
        x_test_emd = joblib.load(os.path.join(texts_path, 'test.texts.embedd'))
        x_train = joblib.load(os.path.join(texts_path, 'train.texts'))
        x_train_emd = joblib.load(os.path.join(texts_path, 'train.texts.embedd'))

    print ("get_pca_traces","......"*8)
    _, pca_data, softmax, pred_seq_labels, train_pred_labels, train_truth_labels = first_abst.classifier.get_pca_traces(args.pca,
                                                                                                   args.epoch)
    print ("get_test_pca_traces","......"*8)
    test_pca_data, test_softmax, test_seq_labels, test_pred_labels, test_truth_labels = first_abst.classifier.get_test_pca_traces(
        args.pca, args.pca, args.epoch)
    
    print ("extract_feature","......"*8)
    train_trace, test_trace = extract_feature(first_abst.ast_model, first_abst.classifier, args.pca, args.epoch, pca_data, pred_seq_labels,
                                              test_pca_data, test_seq_labels, gmm=args.components)



    if args.benign == 0:
        # target_test_indexes = np.where(np.logical_and(test_pred_labels != test_truth_labels, test_truth_labels==1))[0]
        target_test_indexes = np.where(np.logical_and(test_pred_labels != test_truth_labels, test_truth_labels==0))[0]
    else:
        target_test_indexes = np.where(test_pred_labels == test_truth_labels)[0]
        target_test_indexes = np.random.choice(target_test_indexes, 200, replace=False)


    target_test_indexes = target_test_indexes #[0:10] #[122, 142]

    log = ''
    print(len(target_test_indexes))
    #
    failed_text = []
    failed_text_emd = []
    failed_trace = []
    failed_test_pred = test_pred_labels[target_test_indexes]
    failed_test_truth = test_truth_labels[target_test_indexes]
    for i in target_test_indexes:
        failed_text.append(x_test[i])
        failed_text_emd.append(x_test_emd[i])
        failed_trace.append(test_trace[i])

    rand_results = []
    guide_results = []

    rand_times = int(args.rand)


    if args.model =="torch_gru_toxic":
        
        _DynamicClassifier = TOXICClassifier
        _DynamicList = [5, 15, 25, 35, 45]
    elif args.model =="torch_lstm_imdb":
        _DynamicClassifier = IMDBClassifier
        
    elif args.model =="torch_gru_sst":
        _DynamicClassifier = SSTClassifier
        # _DynamicList=[10, 25, 35, 45,75, 95, 105]
        # _DynamicList = [5, 15, 25, 35, 45]
        # _DynamicList = [5, 15, 25, 35, 45,75,85,105]
        _DynamicList = [5, 15, 25, 35, 45,55,65,75,85,95,105,115]
        # _DynamicList = [85]

    else:
        raise Exception("not support now")

    
    tmp_save_dir= os.path.join(save_dir,"tmp")
    os.makedirs(tmp_save_dir, exist_ok=True)
    
    # for sen_num in [5, 15, 25, 35, 45, 80, 100]:
    for sen_num in _DynamicList:

        acc1 = 0
        acc2 = 0
        for _ in range(rand_times):
            temp_train = copy.deepcopy(x_train)
            log, target_test, new_train = repair(first_abst, args.pca, args.epoch, args.components, failed_text, failed_test_pred,
                                      failed_test_truth, failed_text_emd, failed_trace, temp_train, train_trace, sen_num, True)

            import time 
            np.savez(f"{tmp_save_dir}/123_{time.time()}.npz",target_test)
            
            save_dir =get_project_path(args.path,args.model)

            classifier = _DynamicClassifier(rnn_type='gru', train_default=False, save_dir=save_dir, epoch=args.epoch)
            # print ("====="*20)
            # print (type(target_test))
            # print ("len",len(target_test))
            # print (type(target_test[0]))
            # print (target_test.shape,"---",target_test.dtype)
            new_model = classifier.train(dataset=(new_train, target_test))
            pred_labels, _, _, _, truth_labels = classifier.eval_test(new_model, target_test)
            acc1 += np.sum(pred_labels == truth_labels) / len(pred_labels)
            pred_labels, _, _, _,  truth_labels = classifier.eval_test(new_model, x_test)
            acc2 += np.sum(pred_labels == truth_labels) / len(pred_labels)
            print(len(pred_labels))


        acc1 /= rand_times
        acc2 /= rand_times
        # rand_results.append((acc1, acc2))
        rand_results.append(acc1)

        temp_train = copy.deepcopy(x_train)
        log, target_test, new_train = repair(first_abst, args.pca, args.epoch, args.components, failed_text,
                                             failed_test_pred,
                                             failed_test_truth, failed_text_emd, failed_trace, temp_train, train_trace,
                                             sen_num, False)
        classifier = _DynamicClassifier(rnn_type='gru', train_default=False, save_dir=save_dir, epoch=args.epoch)
        new_model = classifier.train(dataset=(new_train, target_test))
        pred_labels, _, _, _, truth_labels = classifier.eval_test(new_model, target_test)
        acc1 = np.sum(pred_labels == truth_labels) / len(pred_labels)
        pred_labels, _, _, _, truth_labels = classifier.eval_test(new_model, x_test)
        acc2 = np.sum(pred_labels == truth_labels) / len(pred_labels)
        # guide_results.append((acc1, acc2))
        guide_results.append(acc1)


    print("Original Acc:" , np.sum(test_pred_labels == test_truth_labels )/len(test_truth_labels))
    print("Random Results:", rand_results)
    print("Guided Results:", guide_results)

