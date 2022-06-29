from sklearn.decomposition import PCA
import os
import numpy as np
import joblib
class Reduction(object):
    def do_reduction(self, new_data):
        pass
class PCA_R(Reduction):


    def __init__(self, top_k):
        self.top_components = top_k
        self.pca = None

    def create_pca(self, data_list):

        # pca_path = os.path.join(dir, str(self.top_components)+'.pca')
        assert(len(data_list) > 0)
        if self.top_components >= data_list[0].shape[-1]:
            self.pca = None
            return data_list, np.amin(data_list, axis=0), np.amax(data_list, axis=0)
        else:
            self.pca = PCA(n_components=self.top_components, copy=False)
            data = np.concatenate(data_list, axis=0)
            # total = []
            # for np_st_vec in data_list:
            #     # np_st_vec: batch_num * length * hidden_length
            #     x, y, z = np_st_vec.shape
            #     total.append(np_st_vec.reshape(x*y, z))

            # data = np.concatenate(total,axis=0)

            ori_pca_data = self.pca.fit_transform(data)

            min_val = np.amin(ori_pca_data, axis=0)
            max_val = np.amax(ori_pca_data, axis=0)

            pca_data = []
            indx = 0

            for state_vec in data_list:
                l = state_vec.shape[0]
                pca_data.append(ori_pca_data[indx: (indx + l)])
                indx += l

            # for np_st_vec in data_list:
            #     for batch in np_st_vec:
            #         l = batch.shape[0]
            #         pca_data.append(ori_pca_data[indx: (indx + l)])
            #         indx += l
            return pca_data, min_val, max_val


    def do_reduction(self, data_list):
        if self.pca is None:
            return data_list
        else:
            # total = []
            # for np_st_vec in data_list:
            #     # np_st_vec: batch_num * length * hidden_length
            #     x, y, z = np_st_vec.shape
            #     total.append(np_st_vec.reshape(x*y, z))
            data = np.concatenate(data_list, axis=0)

            pca_data = self.pca.transform(data)

            new_data = []
            indx = 0
            for state_vec in data_list:
                l = state_vec.shape[0]
                new_data.append(pca_data[indx: (indx + l)])
                indx += l
            # for np_st_vec in data_list:
            #     for batch in np_st_vec:
            #         l = batch.shape[0]
            #         new_data.append(pca_data[indx: (indx + l)])
            #         indx += l

            return new_data










