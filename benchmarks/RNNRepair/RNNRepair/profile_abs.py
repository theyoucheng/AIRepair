from abc import abstractmethod
import os
import joblib
import time
from . import reproducible


def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Profiling(object):
    def __init__(self, save_dir):
        
        if save_dir:
            self.save_dir = save_dir
            self.model_dir = os.path.join(save_dir, 'model')
            self.con_trace_dir = os.path.join(save_dir, 'con_trace')
            self.pca_dir = os.path.join(save_dir, 'pca_trace')
            self.abst_dir = os.path.join(save_dir, 'abs_model')
            self.gan_dir = os.path.join(save_dir, 'feature_data')
            self.input_dir = os.path.join(save_dir, '_texts')
    
            check_mkdir(self.model_dir)
            check_mkdir(self.con_trace_dir)
            check_mkdir(self.pca_dir)
            check_mkdir(self.abst_dir)
            check_mkdir(self.save_dir)
            check_mkdir(self.input_dir)
            check_mkdir(self.input_dir)

    '''
        It is better to cache the profile data in the self.con_trace_dir
    '''
    def preprocess(self,x_test):
        return x_test
    @abstractmethod
    def do_profile(self, test):
        pass
    @abstractmethod
    def get_labels(self):
        pass
    def build_abstract_model(self, k, m, epoch, abst, class_num):
        ast_path = os.path.join(self.abst_dir, str(k)+'_'+str(epoch)+'_'+str(m)+'_'+abst+'.ast')
        pca_path = os.path.join(self.pca_dir, str(k)+'_'+str(epoch) + '.pca')
        # texts_path = os.path.join(self.input_dir, 'train.texts')

        model_exsits = os.path.exists(ast_path)
        pca_exists =  os.path.exists(pca_path)

        if not model_exsits or not pca_exists:

            pca, pca_data, softmax, labels,_,_ = self.get_pca_traces(k, epoch)
            if not model_exsits:
                if abst == 'GMM':
                    from RNNRepair.abstraction.abstraction import GMM
                    ast_model = GMM([pca_data], m, labels, class_num)
                else:
                    print('Not implemented')
                    assert (False)
                joblib.dump(ast_model, ast_path)
        if model_exsits:
            print('Find the cached abstract model', ast_path)
            start = time.time()
            ast_model = joblib.load(ast_path)
            # print('Load the model', time.time() - start)
            if pca_exists:
                pca = joblib.load(pca_path)

        return pca, ast_model

    def get_test_pca_traces(self, pca, k, epoch):
        k_path = os.path.join(self.pca_dir, str(k) + '_' + str(epoch) + '.tptr')
        if os.path.exists(k_path):
            print('found', k_path)
            return joblib.load(k_path)
        else:
            pred_labels, seq_labels, softmax, con_tr, truth_labels = self.get_con_trace(k, epoch, test=True)
            # x_pca_data = pca.do_reduction(con_tr)
            # from RNNRepair.abstraction.reduction import PCA_R
            # pca = PCA_R(k)
            x_pca_data = pca.do_reduction(con_tr)

            joblib.dump((x_pca_data, softmax, seq_labels, pred_labels, truth_labels), k_path)
        return x_pca_data, softmax, seq_labels, pred_labels, truth_labels


    def get_pca_traces(self, k, epoch):
        k_path = os.path.join(self.pca_dir, str(k) +'_'+str(epoch)+ '.ptr')
        pca_path = os.path.join(self.pca_dir, str(k) +'_'+str(epoch)+ '.pca')

        if os.path.exists(k_path):

            pca_data, softmax, seq_labels, final_labels, truth_labels = joblib.load(k_path)
            pca = joblib.load(pca_path)

        else:
            print('Cannot find the PCA traces, we will conduct the PCA reduction')
            final_labels, seq_labels, softmax, con_tr, truth_labels = self.get_con_trace(k, epoch, test=False)

            from RNNRepair.abstraction.reduction import PCA_R
            pca = PCA_R(k)
            pca_data, min_val, max_val = pca.create_pca(con_tr)

            joblib.dump(pca, pca_path)
            joblib.dump((pca_data, softmax, seq_labels, final_labels, truth_labels), k_path)
        return pca, pca_data, softmax, seq_labels, final_labels, truth_labels
    def get_con_trace(self, k, epoch, test):
        if test:
            con_path = os.path.join(self.con_trace_dir, str(k) +'_'+str(epoch)+'.tctr')
        else:
            con_path = os.path.join(self.con_trace_dir, str(k) +'_'+str(epoch)+ '.ctr')
        if os.path.exists(con_path):
            con_tr = joblib.load(con_path)
        else:
            con_tr = self.do_profile(test)
            joblib.dump(con_tr, con_path)
        return con_tr
    
    # def cal_pca_trace(self, pca, x_test, texts = True):
        # pred_labels, seq_labels, softmax, con_tr, truth_labels, texts, texts_emd  = self.eval_test(self.model, x_test, save_text=None, save_name = '', need_loss = False, need_texts=texts)
        # x_pca_data = pca.do_reduction(con_tr)
        # return x_pca_data, softmax, seq_labels, pred_labels, truth_labels, texts, texts_emd
