from collections import OrderedDict
import numpy as np
from graphviz import Digraph
from sklearn.mixture import GaussianMixture
import time
class AbstractModel():
    def __init__(self):
        self.states = []
        self.initial = 0
        transitions = None
        self.final = []
class GMM(AbstractModel):

    def __init__(self, traces, components, labels, class_num):
        super().__init__()
        ts = traces[0]
        self.gmm = GaussianMixture(n_components=components, covariance_type='diag')

        # print(ts.shape, labels.shape)
        new_array = np.concatenate(ts, axis = 0)
        labels = np.concatenate(labels, axis = 0)
        # new_array = new_array.reshape((new_array.shape[0] * new_array.shape[1]), new_array.shape[2])
        print(new_array.shape, labels.shape)
        gmm_labels = self.gmm.fit_predict(new_array)
        gmm_labels += 1

        self.bic = self.gmm.bic(new_array)
        self.m = components + 1
        self.avg, self.max_val,self.min_val = self.evaluate_labels(ts, gmm_labels, labels, class_num)
        self.sort_indexes = np.argsort(self.lbs, axis=-1)
        print('converged: ',self.gmm.converged_)

    def evaluate_labels(self, traces, gmm_labels, labels, cls_num):

        self.lbs = np.zeros((self.m, cls_num),dtype=int)
        self.lbs[0][0] = 1

        idx = 0
        for i, ts in enumerate(traces):
            for j in range(ts.shape[0]):
                cur_gmm_label = gmm_labels[idx]
                cur_pred_label = labels[idx]
                self.lbs[cur_gmm_label][cur_pred_label] += 1
                idx += 1
        self.ratio = self.lbs.max(axis=1)/np.sum(self.lbs, axis=1)
        return np.average(self.ratio), self.ratio.max(), self.ratio.min()
    def update_transitions(self, features, cls_num, abstract_inputs):

        max_id = 0 if abstract_inputs is None else max([max(y) for y in abstract_inputs])
        # self.trans = np.zeros((self.m, self.m), dtype=int)
        transitions = np.zeros((self.m, cls_num, max_id + 1, cls_num, self.m), dtype=int)
        tran_to_training_without = dict()
        tran_to_training = dict()

        for i, feature in enumerate(features):
            gmm_labels = feature[:,0]
            assert (np.sum(gmm_labels == 0) == 0)
            labels = feature[:,1]
            pre_gmm_label = 0
            pred_label = 0

            for j in range(len(feature)):
                cur_gmm_label = int(gmm_labels[j])
                cur_pred_label = int(labels[j])
                emb_num = 0 if abstract_inputs is None else abstract_inputs[i][j]
                # print(gmm_labels.shape, labels.shape, pre_gmm_label,pred_label,emb_num ,cur_pred_label,cur_gmm_label)
                transitions[pre_gmm_label][pred_label][emb_num][cur_pred_label][cur_gmm_label] += 1
                key = (pre_gmm_label, pred_label, emb_num, cur_pred_label, cur_gmm_label)
                if key in tran_to_training:
                    tran_to_training[key].add(i)
                else:
                    tran_to_training[key] = {i}

                key2 = (pre_gmm_label, pred_label, cur_pred_label, cur_gmm_label)
                if key2 in tran_to_training_without:
                    tran_to_training_without[key2].add(i)
                else:
                    tran_to_training_without[key2] = {i}


                pre_gmm_label = cur_gmm_label
                pred_label = cur_pred_label
        return transitions, tran_to_training, tran_to_training_without


    def evaluate_labels_1(self, traces, gmm_labels, labels, cls_num, abstract_inputs):

        self.lbs = np.zeros((self.m, cls_num),dtype=int)
        self.lbs[0][0] = 1

        self.max_id = 0  if abstract_inputs is None else max([max(y) for y in abstract_inputs])
        assert(self.max_id != 0)

        # self.trans = np.zeros((self.m, self.m), dtype=int)
        transitions = np.zeros((self.m, cls_num, self.max_id+1, cls_num, self.m), dtype=int)
        self.tran_to_training = dict()

        idx = 0
        for i, ts in enumerate(traces):
            pre_gmm_label = 0
            pred_label = 0
            for j in range(ts.shape[0]):
                cur_gmm_label = gmm_labels[idx]
                cur_pred_label = labels[idx]
                emb_num = 0 if abstract_inputs is None else abstract_inputs[i][j]
                self.lbs[cur_gmm_label][cur_pred_label] += 1

                # print(transitions.shape, pre_gmm_label,pred_label,cur_pred_label,cur_gmm_label)
                transitions[pre_gmm_label][pred_label][emb_num][cur_pred_label][cur_gmm_label] += 1
                key  = (pre_gmm_label, pred_label, emb_num, cur_pred_label, cur_gmm_label)
                if key in self.tran_to_training:
                    self.tran_to_training[key].add(i)
                else:
                    self.tran_to_training[key] = set([i])
                # self.trans[pre_gmm_label][cur_gmm_label] += 1
                idx += 1
                pre_gmm_label = cur_gmm_label
                pred_label = cur_pred_label

        self.ratio = self.lbs.max(axis=1)/np.sum(self.lbs, axis=1)


        return np.average(self.ratio), self.ratio.max(), self.ratio.min()


    def draw_states(self, path):
        g = Digraph('G', filename=path, engine='sfdp', format='png')
        g.attr('node', shape='box', color='red')
        for i, indexes in enumerate(self.sort_indexes):
            label = str(i) + '\n'
            for j in indexes[::-1]:
                label += str(j) + ', ' + str(self.lbs[i][j]) + ', ' + str(
                    round(self.lbs[i][j] / np.sum(self.lbs[i]), 3)) + '\n'
            g.node(name=str(i), label=label)
        return g


    def evaluate_input (self, trace, pred_labels):
        labels = self.gmm.predict(trace) + 1
        num = self.lbs[labels]

        re = []

        for i, j in enumerate(pred_labels):
            re.append(round(num[i][j]/np.sum(num[i]),4))
        return re


    def get_trace_input(self, train_trace, pred_labels, trans_include=False, only_confidence_include=False):
        results = []
        if trans_include:
            self.trans_ratio = self.trans / np.sum(self.trans)
        preds = np.array(self.lbs / self.lbs.sum(axis=-1)[:,None])
        gmm_labels = []

        trans_res = []

        for k, trace in enumerate(train_trace):
            labels = self.gmm.predict(trace) + 1
            ratios = preds[labels]
            if only_confidence_include:
                ratios = np.expand_dims(ratios[np.arange(len(ratios)),pred_labels[k]], axis=-1)
            results.append(ratios)
            gmm_labels.append(labels)
            if trans_include:
                trans = [0]
                pre = -1
                for i in labels:
                    cur = i
                    if pre != -1:
                        trans.append(self.trans_ratio[pre][cur])
                    pre = cur
                trans_res.append(trans)


        total = []
        for i, item in enumerate(gmm_labels):
            item = np.expand_dims(item, axis=-1)
            pred = np.expand_dims(pred_labels[i], axis=-1)
            temp = np.append(item, pred, axis=-1)
            temp = np.append(temp, results[i], axis=-1)
            total.append(temp)
        return total

    def show_trans_prob(self, pre_st, pre_lbl, tar_lbl, tar_st, word, emd, transitions):
        # state_total = np.sum(transitions[pre_st])
        # state_trans_sum = np.sum(transitions[pre_st], axis=(0,1,2))[tar_st]

        if transitions is None:
            label_total = 0
            label_trans_sum = 0
            emd_trans_sum = 0
            emd_total = 0
        else:
            label_total = np.sum(transitions[pre_st][pre_lbl])
            label_trans_sum = np.sum(transitions[pre_st][pre_lbl], axis=0)[tar_lbl][tar_st]

            emd_trans_sum = transitions[pre_st][pre_lbl][emd][tar_lbl][tar_st]
            emd_total = np.sum(transitions[pre_st][pre_lbl][emd])

        emd_ratio = 'nil' if emd_total == 0 else str(round(emd_trans_sum / emd_total, 2))
        # st_ratio = 'nil' if state_total == 0 else str(round(state_trans_sum/state_total, 2))
        lbl_ratio = 'nil' if label_total == 0 else str(round(label_trans_sum / label_total, 2))
        # tag = ' --[' + str(state_trans_sum)+'/' +str(state_total)+', '+st_ratio+']['+str(label_trans_sum)+'/'+str(label_total)+ ', '+lbl_ratio+']-> '
        # tag = ' == '+word + ' {' + st_ratio + ',' + lbl_ratio + ','+ emd_ratio+'} => '
        tag = ' == ' + word + ' {' + lbl_ratio + ',' + emd_ratio + '} => '
        return tag


    def text_output(self, abst_traces, probs, word, word_emd, indexes, transitions):
        trace_str = ''
        if word_emd is None:

            for j, trace in enumerate(abst_traces):
                abst_states = trace[:, 0].astype(int)  # self.gmm.predict(trace)
                pred = 0
                pred_labels = trace[:, 1].astype(int)
                trace_str += ('\r\n============' + str(word[indexes[j]]) + '  Trace===========\r\n')

                total_len = len(abst_states)

                for i, lb in enumerate(abst_states):

                    if pred == 0:
                        trace_str += 'Init' + self.show_trans_prob(0, 0, pred_labels[i], lb, '', 0, transitions)

                    trace_str += str(lb) + '[' + str(pred_labels[i]) + ' : ' + str(
                        round(probs[j][i], 2)) + ']'

                    if i + 1 < total_len:
                        trace_str += self.show_trans_prob(lb, pred_labels[i], pred_labels[i + 1], abst_states[i + 1],
                                                          '', 0, transitions)
                    if i % 5 == 0 and i > 0:
                        trace_str += '\n'
                    pred = lb
            return trace_str

        else:
            for j, trace in enumerate(abst_traces):
                abst_states = trace[:, 0].astype(int)  # self.gmm.predict(trace)
                pred = 0
                input_w = word[j][1]
                input_emd = word_emd[j]

                pred_labels = trace[:, 1].astype(int)
                # print(len(input_w), len(pred_labels))
                assert (len(input_w) == len(pred_labels))
                trace_str += ('\r\n============' + str(indexes[j])+'  Text===========\r\n')
                trace_str += 'GroundTruth (Pos): ' if word[j][0] == True else 'GroundTruth (Neg): '
                trace_str += ' '.join(input_w)
                trace_str += '\r\n============Trace===========\r\n'

                total_len = len(abst_states)

                for i, lb in enumerate(abst_states):

                    if pred == 0:
                        trace_str += 'Init' + self.show_trans_prob(0, 0, pred_labels[i], lb, input_w[0], input_emd[0],transitions)

                    trace_str += str(lb) + '[' + str(pred_labels[i]) + ' : ' + str(
                        round(probs[j][i], 2)) + ']'

                    if i + 1 < total_len:
                        trace_str += self.show_trans_prob(lb, pred_labels[i], pred_labels[i + 1], abst_states[i + 1],
                                                          input_w[i + 1], input_emd[i + 1],transitions)
                    if i % 5 == 0 and i > 0:
                        trace_str += '\n'
                    pred = lb
                # trace_str += '\n\n'
            return trace_str


    def visualize_inputs(self, path, abst_traces, probs, names, word, word_emd,transitions):

        g = Digraph('G', filename=path, format='png')
        g.attr('node', shape='box', color='red')
        colors = ['blue', 'orange', 'black', 'cyan']

        state_info = {}

        for i, indexes in enumerate(self.sort_indexes):
            label = str(i) + '\n'
            for j in indexes[::-1]:
                label += str(j) + ', ' + str(self.lbs[i][j]) + ', ' + str(
                    round(self.lbs[i][j] / np.sum(self.lbs[i]), 3)) + '\n'
            state_info[i] = label
            # g.node(name=str(i), label=label)
        trace_str = ''

        # add start
        g.node(name = '0',label = state_info[0])

        if word_emd is None:
            for j, trace in enumerate(abst_traces):
                abst_states = trace[:, 0].astype(int)  # self.gmm.predict(trace)
                pred = 0


                g.attr('edge', color=colors[j % len(colors)])

                pred_labels = trace[:, 1].astype(int)

                trace_str += names[j] + ' : '

                total_len = len(abst_states)

                for i, lb in enumerate(abst_states):

                    if pred == 0:
                        trace_str += 'Init' + self.show_trans_prob(0, 0, pred_labels[i], lb, '', 0,
                                                                   transitions)

                    trace_str += str(lb) + '[' + str(pred_labels[i]) + ' : ' + str(
                        round(probs[j][i], 2)) + ']'

                    if i + 1 < total_len:
                        trace_str += self.show_trans_prob(lb, pred_labels[i], pred_labels[i + 1], abst_states[i + 1],
                                                          '', 0, transitions)
                    if i % 5 == 0 and i > 0:
                        trace_str += '\n'
                    g.node(name = str(lb), label = state_info[lb])
                    g.edge(str(pred), str(lb))
                    pred = lb
                trace_str += '\n\n'

                g.attr(label=trace_str)
        else:

            for j, trace in enumerate(abst_traces):
                abst_states= trace[:,0].astype(int)#self.gmm.predict(trace)
                pred = 0

                input_w = word[j]
                input_emd = word_emd[j]


                g.attr('edge', color=colors[j % len(colors)])


                pred_labels = trace[:, 1].astype(int)
                assert (len(input_w) == len(pred_labels))

                trace_str += names[j] + ' : '

                total_len = len(abst_states)

                for i, lb in enumerate(abst_states):

                    if pred == 0:

                        trace_str += 'Init' + self.show_trans_prob(0, 0, pred_labels[i], lb, input_w[0], input_emd[0],transitions)


                    trace_str += str(lb) + '[' + str(pred_labels[i]) + ' : ' + str(round(probs[j][i], 2)) + ']'

                    if i + 1 < total_len:

                        trace_str += self.show_trans_prob(lb, pred_labels[i], pred_labels[i+1], abst_states[i+1],input_w[i+1], input_emd[i+1],transitions)
                    if i % 5 == 0 and i > 0:
                        trace_str += '\n'
                    g.node(name=str(lb), label=state_info[lb])
                    g.edge(str(pred), str(lb))
                    pred = lb
                trace_str += '\n\n'

                g.attr(label=trace_str)
        return g


    def visulize_input(self, path, trace, pred_labels, probabilities):

        # labels = np.argmax(self.lbs, axis=-1)
        # max_lbs = self.lbs.max(axis=1)
        g = Digraph('G', filename=path, engine='sfdp', format='png')
        g.attr('node', shape='box', color='red')

        for i, indexes in enumerate(self.sort_indexes):
            label = str(i) + '\n'
            for j in indexes[::-1]:
                label+= str(j) +', '+ str(self.lbs[i][j])+', '+str(round(self.lbs[i][j]/np.sum(self.lbs[i]), 3))+'\n'
            g.node(name=str(i), label=label)

        labels = self.gmm.predict(trace) + 1
        trace = ''
        pred = None
        for i, lb in enumerate(labels):

            trace += str(lb) + '['+str(pred_labels[i])+' : '+str(round(probabilities[i][pred_labels[i]],2))+'] -> '
            if i%10 == 0:
                trace += '\n'
            if pred is None:
                pred = lb
                continue
            else:
                g.edge(str(pred), str(lb))
                pred = lb

        g.attr(label=trace)
        return g





    def state_abstract(self, con_state, gmm):
        return gmm.predict()

    def info_trace(self, trace):
        results = []
        for gmm in self.gmms:
            results.append(gmm.predict(trace))
        return  results

