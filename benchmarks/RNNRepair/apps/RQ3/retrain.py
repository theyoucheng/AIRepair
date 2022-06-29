import sys
# sys.path.append("../../")
from keras.datasets import mnist
import numpy as np
import os
import joblib
import torch
from collections import defaultdict

from RNNRepair.use_cases.image_classification.mnist_rnn_profile_torch import RNN,TorchMnistiClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')
    parser.add_argument('-save_dir', default="./save/", type=str)
    
    parser.add_argument('-epoch', default=30, type=int)
    parser.add_argument('-flipfirst', default=1, type=int)
    parser.add_argument('-flipsecond', default=7, type=int)
    parser.add_argument('-flip', default=2, type = int)
    parser.add_argument('-ratio', default=0.3, type=float)
    parser.add_argument('-start_seed', default=1000, type=int)
    parser.add_argument('-end_seed', default=1010, type=int)
    
    args = parser.parse_args()
    save_dir =args.save_dir 
    

    epoch= args.epoch
    flipfirst = args.flipfirst
    flipsecond = args.flipsecond
    flip_mode = args.flip
    ratio = args.ratio
    start_seed = args.start_seed
    end_seed = args.end_seed

    seeds = np.arange(start_seed,end_seed)
    path = "dataflip_{}_{}_{}".format(flipfirst,flipsecond,ratio)
    path = os.path.join(save_dir,path)
    
    
    parts = 10
    size = 1/parts
    all_test_results = defaultdict()
    all_target_results = defaultdict()
    all_results = defaultdict()

    # seeds = [1]

    log = open("{}/retrain_{}_{}_{}_{}.log".format(path,flipfirst,flipsecond,flip_mode,ratio),"w+")
    for cur_seed in seeds:
        flip_save_dir = os.path.join(save_dir, 'torch_lstm_bin', str(cur_seed)+'_'+str(flip_mode))
        classifier = TorchMnistiClassifier(rnn_type='lstm', save_dir=flip_save_dir, train_default=False, 
                                        epoch=epoch, seed=cur_seed, flip=flip_mode, first=flipfirst,second=flipsecond, ratio = ratio)

        (x_train,y_train),(x_test,y_test),flip_idx = classifier.load_binary_data()
        real_flip_idx = np.load("./{}/infl_flip_idx({}_{}_{}_{}_{}).npy".format(path,flipfirst,flipsecond,flip_mode,ratio,cur_seed))
        target_test_idx = np.load("./{}/target_test_{}_{}_{}_{}_{}.npy".format(path,flipfirst,flipsecond,flip_mode,ratio,cur_seed))
        
        print("[load]cur_seed:{}\t\tflipidx:{}\t\tinfl_flip_idx:{}\t\ttarget_test:{}".format(cur_seed,len(flip_idx),len(real_flip_idx),len(target_test_idx)))

        sgd_order = joblib.load(os.path.join(save_dir,
                     "./mnist_rnn_sgd_{}_{}_{}_{}/mnist_rnn_sgd_{:02d}/infl_sgd_at_epoch{}_00.dat".format(
            flipfirst,flipsecond,flip_mode,ratio,cur_seed,epoch)
                     )
                     )[:,-1]
        icml_order = joblib.load(os.path.join(save_dir,
                      "./mnist_rnn_sgd_{}_{}_{}_{}/mnist_rnn_sgd_{:02d}/infl_icml_at_epoch{}_00.dat".format(
            flipfirst,flipsecond,flip_mode,ratio,cur_seed,epoch)
                      )
                      )
        jcard_order = np.load(os.path.join(save_dir,
                       "./dataflip_{}_{}_{}/jcard_order_{}_{}_{}_{}_{}.npy".format(
            flipfirst,flipsecond,ratio,flipfirst,flipsecond,flip_mode,ratio,cur_seed))
                       )
        
        all_results[cur_seed] = defaultdict()
        all_results[cur_seed]['sgd'] = np.argsort(sgd_order)
        all_results[cur_seed]['icml'] = np.argsort(icml_order)
        all_results[cur_seed]['jcard'] = jcard_order
        all_results[cur_seed]['random'] = np.setdiff1d(np.arange(len(sgd_order)),real_flip_idx) 
        np.random.shuffle(all_results[cur_seed]['random'])
        all_results[cur_seed]['random'] = list(all_results[cur_seed]['random']) + list(real_flip_idx)
        all_results[cur_seed]['random'] = np.array(all_results[cur_seed]['random'])
        
        
        all_test_results[cur_seed] = defaultdict()
        all_target_results[cur_seed] =  defaultdict() 
        
        for k in all_results[cur_seed].keys():

            cur_order = np.array(all_results[cur_seed][k]).squeeze()
            all_test_results[cur_seed][k] = []
            all_target_results[cur_seed][k] = []
            _,test_acc,target_acc = classifier.retrain(None, False, None, None,target_test_idx)
            if target_acc != 0:
                print("!!!skip this seed:{}".format(cur_seed))
                log.write("!!!skip this seed:{}\n".format(cur_seed))
                break

            all_test_results[cur_seed][k].append(test_acc)
            all_target_results[cur_seed][k].append(target_acc)
            print("[{}]No restore results\t\tcur_seed:{}\t\ttest_acc:{}\t\ttarget_acc{}\n".format(k,cur_seed,test_acc,target_acc))
            print("##############################################")
            log.write("[{}]No restore results\t\tcur_seed:{}\t\ttest_acc:{}\t\ttarget_acc{}\n".format(k,cur_seed,test_acc,target_acc))
            log.write("##############################################\n")
            for i in range(parts):
                cur_part = cur_order[:int(len(cur_order)*size*(i+1))]
                cur_restore_idx = list(set(cur_part).intersection(set(flip_idx)))
                print(">>>>>[{}]restore top {:.2f} ,cur_seed:{}, find {}/{} restore idx".format(k,size*(i+1),cur_seed,len(cur_restore_idx),len(flip_idx)))
                log.write(">>>>>[{}]restore top {:.2f} ,cur_seed:{}, find {}/{} restore idx\n".format(k,size*(i+1),cur_seed,len(cur_restore_idx),len(flip_idx)))
                
                if len(cur_restore_idx)==0:
                    print("[{}]No restore idx,continue;".format(k))
                    log.write("[{}]No restore idx,continue;\n".format(k))
                    all_test_results[cur_seed][k].append(all_test_results[cur_seed][k][-1])
                    all_target_results[cur_seed][k].append(all_target_results[cur_seed][k][-1])
                    continue
                else:
                    # removed_idx=None, is_load = False, saved_path = None, restore_idx=None,target_test_idx=None
                    _,test_acc,target_acc = classifier.retrain(None, False, None,cur_restore_idx,target_test_idx)
                    print("<<<<[{}]cur_seed:{},restore top {:.2f} , find {}/{} restore idx: {},{}".format(
                        k,cur_seed,size*(i+1),len(cur_restore_idx),len(flip_idx),test_acc,target_acc))
                    log.write("<<<<[{}]cur_seed:{},restore top {:.2f} , find {}/{} restore idx: {},{}\n".format(
                        k,cur_seed,size*(i+1),len(cur_restore_idx),len(flip_idx),test_acc,target_acc))
                    
                    all_test_results[cur_seed][k].append(test_acc)
                    all_target_results[cur_seed][k].append(target_acc)
            log.write("\n\n")
            log.flush()
        log.write(">>>>>>>>>>>Finish cur seed:{} for [{}]\n\n\n\n".format(cur_seed,k))
        log.flush()
        
    log.close()

    saved_results = {"all_results":all_results,"all_test_results":all_test_results,"all_target_results":all_target_results}
    joblib.dump(saved_results,"{}/retrain_results_{}_{}_{}_{}.dat".format(path,flipfirst,flipsecond,flip_mode,ratio))


    # saved_results = joblib.load("{}/all_retrain_results.dat".format(path))
    # all_results = saved_results['all_results']
    # all_test_results = saved_results['all_test_results']
    # all_target_results = saved_results['all_target_results']
    # seed = 1

    avg_test_results = defaultdict()
    all_test = defaultdict()

    avg_target_results = defaultdict()
    all_target = defaultdict()

    for method in all_results[cur_seed].keys():
        avg_test_results[method] = np.array([0.0]*(parts+1))
        all_test[method] = []

        avg_target_results[method] = np.array([0.0]*(parts+1))
        all_target[method] = []
        
    for cur_seed in seeds:
        for key in all_results[cur_seed].keys():
            if cur_seed in all_test_results and  key in  all_test_results[cur_seed] and  all_test_results[cur_seed][key]:
                avg_test_results[key] += np.array(all_test_results[cur_seed][key])
                all_test[key].append(np.array(all_test_results[cur_seed][key]))
            
                avg_target_results[key] += np.array(all_target_results[cur_seed][key])    
                all_target[key].append(np.array(all_target_results[cur_seed][key]))
            
    for method in all_results[cur_seed].keys():
        avg_test_results[method] /= parts
        avg_target_results[method] /= parts
        
        all_test[method] = np.array(all_test[method])
        all_target[method] = np.array(all_target[method])


    saved_results={
        "all_results":all_results,
        "all_test_results":all_test_results,"all_target_results":all_target_results,
        "avg_test_results":avg_test_results,"avg_target_results":avg_target_results,
        "all_test":all_test,"all_target":all_target
    }
    joblib.dump(saved_results,"{}/retrain_results_{}_{}_{}_{}.dat".format(path,flipfirst,flipsecond,flip_mode,ratio))


