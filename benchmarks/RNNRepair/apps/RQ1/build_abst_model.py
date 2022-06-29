import numpy as np
# import sys
# sys.path.append("../../")
import joblib
import os
import time
import argparse

from RNNRepair.utils import create_args
from RNNRepair.use_cases import create_classifer
from RNNRepair.abstraction.feature_extraction import extract_feature

if __name__ == "__main__":

    parser  = create_args()
    args = parser.parse_args()


    # save_dir = os.path.join(args.path,  args.model)



    classifier =  create_classifer(model_type=args.model, save_dir=args.path, epoch=args.epoch)
    log_path = os.path.join(classifier.abst_dir,
                            str(args.pca) + '_' + str(args.epoch) + '_s'+str(args.start) +'_abst.log')
    plot_file = open(log_path, 'a+')


    # plot_file.write("index,bic,avg,max,min,time\n")#columns
    print('Start to build abstract models')

    for i in range(args.start, args.end, 3):
        print('Build the GMM with components', i)
        start = time.time()
        pca, ast_model = classifier.build_abstract_model(args.pca, i, args.epoch, 'GMM', classifier.n_classes)
        best = ast_model.bic

        plot_file.write(
            "%d,%d,%f,%f,%f,%d\n" %
            (i,
             ast_model.bic,
             ast_model.avg,
             ast_model.max_val,
             ast_model.min_val,
             time.time() - start
             ))
        plot_file.flush()
    plot_file.close()

    print('Finish building abstract models')




    print('Start to extract features of all training data')


    pca, pca_data, softmax, pred_seq_labels, pred_labels, train_labels = classifier.get_pca_traces(args.pca,
                                                                                                   args.epoch)
    test_pca_data, test_softmax, test_seq_labels, test_pred_labels, test_truth_labels = classifier.get_test_pca_traces(
        pca, args.pca, args.epoch)


    indexes = np.where(pred_labels == train_labels)[0]

    for stnum in range(args.start, args.end, 3):
        print('Start build trace for', stnum)
        path = os.path.join(classifier.abst_dir, 
                            str(args.pca)+'_'+str(args.epoch)+\
                            '_'+str(stnum)+'_GMM.ast')

        if not os.path.exists(path):
            print('Cannot find', path)
            continue
        best_model = joblib.load(path)



        train_trace, test_trace = extract_feature(best_model, classifier, args.pca, args.epoch, pca_data, pred_seq_labels,
                                                 test_pca_data, test_seq_labels, gmm=stnum)


