from keras.datasets import mnist
import os
import numpy as np


import sys
# sys.path.append("../../")
import  torch

from RNNRepair.utils import create_args, get_project_path
from RNNRepair.utils import save_image, get_traces,calculate_similarity_list

from RNNRepair.use_cases import create_classifer
from RNNRepair.abstraction.feature_extraction import extract_feature
from RNNRepair.use_cases.image_classification.mutators import Mutators

from RNNRepair.use_cases.image_classification.mnist_rnn_profile_keras import MnistClassifier

import joblib


def find_target_model(classifier, k, epoch, theta=0.72):
    best_path = os.path.join(classifier.abst_dir, str(k) + '_' + str(epoch) + '_best.ast')

    if os.path.exists(best_path):
        print('Loaded existing best abstract model', best_path)
        best_model = joblib.load(best_path)
    else:
        best_model = None

        for i in range(10, sys.maxsize, 3):

            pca, ast_model = classifier.build_abstract_model(k, i, epoch, 'GMM', 10)
            if ast_model.avg >= theta:
                best_model = ast_model
                break

        joblib.dump(best_model, best_path)
    return best_model

def compute_best_img(imgs, classifier, pca, best_model, pred_img_trace):
    new_imgs = np.array(imgs)

    traces, softmax = get_traces(new_imgs, classifier, pca, best_model)
    gen_lbls = np.argmax(softmax, axis=-1)[:,-1]
    return calculate_similarity_list(pred_img_trace, traces, best_model.m), gen_lbls


def data_gen(imgs, pred_trace, classifier, pca, best_model, label):

    rotate_img = list(range(-180, 180))
    trans_img = list(range(-3, 3))

    imgs = np.expand_dims(imgs, axis=-1)

    results = []
    score = []
    labels = []
    new_labels = []
    for img in imgs:
        temp = []
        for i in rotate_img:
            temp.append(Mutators.image_rotation(img, i))
        for i in trans_img:
            temp.append(Mutators.image_translation(img, i))

        rotate_scores, gen_labels = compute_best_img(np.array(temp), classifier, pca, best_model, pred_trace)
        min = np.argmin(rotate_scores)
        results.append(temp[min])
        score.append(rotate_scores[min])
        labels.append(label)
        new_labels.append(gen_labels[min])
    return results, np.array(score), labels, new_labels



if __name__ == "__main__":
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')
    # parser.add_argument('-pca', default=10, type=int)
    # parser.add_argument('-epoch', default=15, type=int)
    # parser.add_argument('-path')
    # parser.add_argument('-components', default=43, type=int)
    #
    # parser.add_argument('-model', default='keras_lstm_mnist',
                        # choices=['keras_lstm_mnist', 'torch_gru_imdb', 'torch_gru_toxic', 'torch_lstm_bin', 'torch_gru_sst', ])
    # args = parser.parse_args()
    args = create_args().parse_args()

    save_dir = os.path.join(args.path, 'data', args.model)


    classifier = create_classifer(model_type=args.model,epoch=args.epoch) 

    K = 1
    #set it for generating the common failed inputs
    pca_models = [15]



    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    tempfiles = os.path.join(save_dir, 'temp_files')
    os.makedirs(tempfiles, exist_ok=True)
    # pca_models = [5,8, 10, 12, 15,18,20]

    str_name = '_'.join(str(x) for x in pca_models)
    failed_test_path = os.path.join(tempfiles+'/failed_'+str_name+'.npy')

    if os.path.exists(failed_test_path):
        indexes = np.load(failed_test_path)
    else:

        # Find the common errors in all models
        x_test = classifier.preprocess(x_test)
        indexes = np.ones(len(y_test), dtype=bool)
        if args.model == 'keras_lstm_mnist' or args.model == 'keras_gru_mnist':
            for m in pca_models:
                m = classifier.rnn_type+'_'+str(m)+'.h5'
                print(m)
                model = classifier.create_model()
                model.load_weights(os.path.join(classifier.model_dir, m))
                prediction = np.argmax(model.predict(x_test), axis=-1)
                indexes = np.logical_and (indexes, (prediction!=y_test))
        else:
            print('Not support now')
            exit()
        np.save(failed_test_path, indexes)


    failed_test_indexes = np.where(indexes==True)[0]

    print('Find common errors', len(failed_test_indexes))


    save_image(tempfiles, x_test[failed_test_indexes], 'all_common_failed_tests')

    # print(failed_test_indexes)
    select_images = []

    save_dir = classifier.save_dir


    retrain_id = failed_test_indexes
    retrain_img = []
    retrain_label = []


    retrain_pred_score = []
    retrain_truth_score = []
    retrain_best_score = []
    retrain_ids = []
    retrain_best_labels = []
    retrain_pred_labels = []
    retrain_truth_labels = []
    retrain_best_pred_labels = []




    rnn_type = classifier.rnn_type
    retrain_name = tempfiles+'/retrain'+'_'.join(str(x) for x in pca_models)+'.npz'
    for m in pca_models:
        m_path = tempfiles+'/remul_'+str(m)+'.job'
        if os.path.exists(m_path):
            all =  joblib.load(m_path)
            m_best_imgs = all[0]
            m_best_labels = all [1]
        else:

            m_best_imgs = []
            m_best_labels = []
            classifier = MnistClassifier(rnn_type=rnn_type, save_dir=save_dir, epoch=m)

            pca, pca_data, softmax, pred_seq_labels, pred_labels, train_labels = classifier.get_pca_traces(args.pca,
                                                                                                           m)
            test_pca_data, test_softmax, test_seq_labels, test_pred_labels, test_truth_labels = classifier.get_test_pca_traces(
                pca, args.pca, m)
            train_pred_results = pred_labels
            # save_dir = os.path.join(get_project_root(), 'data', args.model)
            save_dir = get_project_path(args.path,args.model)

            abst_dir = os.path.join(save_dir, 'abs_model')
            model_path = os.path.join(abst_dir,
                                      str(args.pca) + '_' + str(args.epoch) + '_' + str(args.components) + '_GMM.ast')


            # best_model = find_target_model(classifier, args.pca, m, 0.72)
            best_model = joblib.load(model_path)

            train_trace, test_trace = extract_feature(best_model, classifier, args.pca, m, pca_data,
                                                      pred_seq_labels,
                                                      test_pca_data, test_seq_labels, gmm=best_model.m)



            test_pred_results =  test_pred_labels[failed_test_indexes]



            with torch.no_grad():
                correct_indexes = np.where(pred_labels == train_labels)[0]

                correct_traces = [train_trace[i] for i in correct_indexes]
                correct_imgs = x_train[correct_indexes]

                for i, index in enumerate(failed_test_indexes):
                    print('Generation: use model', m, 'Process failed input', i, index)
                    pred_label = test_pred_results[i]
                    truth_label = y_test[index]

                    similarities = calculate_similarity_list(test_trace[index], correct_traces, best_model.m)

                    sort_indexes = np.argsort(similarities)
                    close_labels = y_train[correct_indexes][sort_indexes]


                    truth_indexes = np.where(close_labels == truth_label)[0]

                    pred_indexes = np.where(close_labels ==  pred_label)[0]


                    truth_results =  round(similarities[sort_indexes[truth_indexes[0]]], 3)
                    pred_results =  round(similarities[sort_indexes[pred_indexes[0]]], 3)

                    cur_img = [x_test[index]]
                    # Line 14 in Algorithm 2
                    p_index = sort_indexes[pred_indexes[0]]

                    #fix
                    bug_img = x_test[index]
                    pred_trace = correct_traces[p_index]

                    # Line 16 in Algorithm 2
                    t_indexes = sort_indexes[truth_indexes[0:K]]

                    gen_fit, gen_img, gen_trace, gen_probs = 1000, None, None, None


                    gen_results, scores, labels, gen_labels = data_gen(correct_imgs[t_indexes], pred_trace, classifier, pca, best_model, truth_label)

                    best_id = np.argmin(scores)

                    best_img = gen_results[best_id]
                    best_score = scores[best_id]

                    #Line 22 in Algorithm 2
                    m_best_imgs.append(best_img)
                    m_best_labels.append(truth_label)

                all = [m_best_imgs, m_best_labels]
                joblib.dump(all, m_path)

        retrain_img.extend(m_best_imgs)
        retrain_label.extend(m_best_labels)

    np.savez(retrain_name, data=np.array(retrain_img), label=np.array(retrain_label), testid = np.array(failed_test_indexes))

    print(len(retrain_label), 'new train data')