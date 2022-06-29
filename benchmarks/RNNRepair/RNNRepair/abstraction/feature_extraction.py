import os
import joblib
import sys
def extract_feature(best_model, classifier, k, epoch, train_pca_data, train_pred_traces, test_pca_data,test_pred_traces, gmm=0, only_confidence=False):
    if only_confidence:
        gan_train = os.path.join(classifier.save_dir, 'feature_data', str(k) +'_'+str(epoch)+ '_'+str(gmm)+'_feature_conf.npz')
    else:
        gan_train = os.path.join(classifier.save_dir, 'feature_data', str(k) +'_' +str(epoch) + '_'+str(gmm)+ '_feature.npz')

    if os.path.exists(gan_train):
        print('found the extracted features', gan_train)
        return joblib.load(gan_train)
    else:
        os.makedirs(classifier.save_dir+'/feature_data', exist_ok=True)

        train_set = best_model.get_trace_input(train_pca_data, train_pred_traces, only_confidence_include=only_confidence)
        test_set = best_model.get_trace_input(test_pca_data, test_pred_traces,only_confidence_include= only_confidence)

        joblib.dump((train_set, test_set), gan_train)

        return train_set, test_set

