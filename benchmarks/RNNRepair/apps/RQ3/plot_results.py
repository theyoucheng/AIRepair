import numpy as np
import sys
import joblib
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.use('Agg')
import seaborn as sns 
sns.set()
sns.set_style("white")


# python plot_results.py 1 7 2 0.3 30 1000 1010
if __name__=="__main__":
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

    path = save_dir 
    # flipfirst = int(sys.argv[1])#1
    # flipsecond = int(sys.argv[2])#7
    # flip_mode = int(sys.argv[3])#2
    # ratio = float(sys.argv[4])#0.3
    # epoch = int(sys.argv[5])#30
    # start_seed= int(sys.argv[6])#1000
    # end_seed = int(sys.argv[7])#1010
    epoch= args.epoch
    flipfirst = args.flipfirst
    flipsecond = args.flipsecond
    flip_mode = args.flip
    ratio = args.ratio
    start_seed = args.start_seed
    end_seed = args.end_seed

    part = 20
    size = 1/part
    

    all_results = defaultdict()
    total_flips_ratio = defaultdict()
    infl_flips_ratio = defaultdict()

    seeds = np.arange(start_seed,end_seed)
    
    for cur_seed in seeds:
        sgd_order = joblib.load("{}/mnist_rnn_sgd_{}_{}_{}_{}/mnist_rnn_sgd_{:02d}/infl_sgd_at_epoch{}_00.dat".format(
            save_dir,flipfirst,flipsecond,flip_mode,ratio,cur_seed,epoch))[:,-1]
        icml_order = joblib.load("{}/mnist_rnn_sgd_{}_{}_{}_{}/mnist_rnn_sgd_{:02d}/infl_icml_at_epoch{}_00.dat".format(
            save_dir,flipfirst,flipsecond,flip_mode,ratio,cur_seed,epoch))
        jcard_order = np.load("{}/dataflip_{}_{}_{}/jcard_order_{}_{}_{}_{}_{}.npy".format(
            save_dir,flipfirst,flipsecond,ratio,flipfirst,flipsecond,flip_mode,ratio,cur_seed))

        random_order = np.arange(len(sgd_order))
        np.random.shuffle(random_order)

        flip_file = "{}/dataflip_{}_{}_{}/torch_lstm_bin/{}_{}/model/flip{}_{}_{}.data".format(
                save_dir,
                flipfirst,flipsecond,ratio,cur_seed,flip_mode,flipfirst,flipsecond,flip_mode)
        (x_train, y_train), (x_test, y_test), flip_idx = joblib.load(flip_file)
        real_flip_idx = np.load("{}/dataflip_{}_{}_{}/infl_flip_idx({}_{}_{}_{}_{}).npy".format(
                            save_dir,

            flipfirst,flipsecond,ratio,flipfirst,flipsecond,flip_mode,ratio,cur_seed))


        all_results[cur_seed] = defaultdict()
        all_results[cur_seed]['sgd'] = np.argsort(sgd_order)
        all_results[cur_seed]['icml'] = np.argsort(icml_order)
        all_results[cur_seed]['jcard'] = jcard_order
        all_results[cur_seed]['random'] = random_order

        total_flips_ratio[cur_seed] = defaultdict()
        infl_flips_ratio[cur_seed] = defaultdict()
        
        for k in all_results[cur_seed].keys():
            cur_order = np.array(all_results[cur_seed][k]).squeeze()
            total_flips_ratio[cur_seed][k] = []
            infl_flips_ratio[cur_seed][k] = []
            for i in range(part):
                cur_part = cur_order[:int(len(cur_order)*size*(i+1))]

                cur_total_flips_common = list(set(cur_part).intersection(set(flip_idx)))
                cur_total_flips_ratio = len(cur_total_flips_common)/len(flip_idx)

                cur_infl_flips_common = list(set(cur_part).intersection(set(real_flip_idx)))
                cur_infl_flips_ratio = len(cur_infl_flips_common)/len(real_flip_idx)

                total_flips_ratio[cur_seed][k].append(cur_total_flips_ratio)
                infl_flips_ratio[cur_seed][k].append(cur_infl_flips_ratio)


    avg_total_flips = defaultdict()
    avg_infl_flips = defaultdict()
    for method in all_results[cur_seed].keys():
        avg_total_flips[method] = np.array([0.0]*part)
        avg_infl_flips[method] = np.array([0.0]*part)

    for cur_seed in seeds:
        for key in total_flips_ratio[cur_seed].keys():
            avg_total_flips[key] += np.array(total_flips_ratio[cur_seed][key])
            avg_infl_flips[key] += np.array(infl_flips_ratio[cur_seed][key])

    for method in all_results[cur_seed].keys():
        avg_total_flips[method] /= len(seeds)
        avg_infl_flips[method] /= len(seeds)
        
    plot_keys = ["Random","K&L" ,"SGD","Proposed"]
    plot_1_data = defaultdict()#;copy.deepcopy(avg_ratio_results)
    plot_1_data["Random"] = avg_total_flips['random']
    plot_1_data["K&L"] = avg_total_flips['icml']
    plot_1_data["SGD"] = avg_total_flips['sgd']
    plot_1_data["Proposed"] = avg_total_flips['jcard']
    
    plot_2_data = defaultdict()
    plot_2_data["Random"] = avg_infl_flips['random']
    plot_2_data["K&L"] = avg_infl_flips['icml']
    plot_2_data["SGD"] = avg_infl_flips['sgd']
    plot_2_data["Proposed"] = avg_infl_flips['jcard']
    
    
    saved_results = joblib.load("{}/dataflip_{}_{}_{}/retrain_results_{}_{}_{}_{}.dat".format(
                        save_dir,
                flipfirst,flipsecond,ratio,flipfirst,flipsecond,flip_mode,ratio))
    print (list(saved_results),"key..saved_results")
    all_results = saved_results['all_results']
    all_target_results = saved_results['all_target_results']
    all_target =  saved_results['all_target']
    avg_target_results = saved_results['avg_target_results']
    parts = 10
    size = 0.1
    
    plot_3_data = defaultdict()
    plot_3_data["Random"] = all_target['random']
    plot_3_data["K&L"] = all_target['icml']
    plot_3_data["SGD"] = all_target['sgd']
    plot_3_data["Proposed"] = all_target['jcard']
    
    plt.figure(figsize=(18,5))
    for idx in range(1,4):
        plt.subplot(1,3,idx)
        for k in plot_keys:
            if idx ==1:
                plt.plot(0.05*np.arange(0,part+1,2),np.insert(plot_1_data[k],0,0)[np.arange(0,part+1,2)],marker=".",label=k)
                plt.ylabel("Fraction of total flips fixed")
            elif idx ==2:
                plt.plot(0.05*np.arange(0,part+1,2),np.insert(plot_2_data[k],0,0)[np.arange(0,part+1,2)],marker=".",label=k)
                plt.ylabel("Fraction of influential flips fixed")
            else:
                plt.plot(0.1*np.arange(11),plot_3_data[k].mean(axis=0)/100,marker=".",label=k)
                plt.ylabel("Fraction of errors repaired")
            plt.xlabel("Fraction of train data checked")
            plt.legend(loc=2)

    plt.savefig("{}/plot_{}_{}_{}.pdf".format(                save_dir,flipfirst,flipsecond,flip_mode),bbox_inches='tight',dpi=300)
    print("save {}/plot_{}_{}_{}.pdf successfully!".format(                save_dir,flipfirst,flipsecond,flip_mode))
            
    
    
    
    