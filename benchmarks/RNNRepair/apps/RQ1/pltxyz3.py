# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import functools
import numpy as np
import os 

here_dir = os.path.dirname(__file__)

from matplotlib import  rcParams

px = {'legend.fontsize': 'x-large',
#          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

rcParams.update(px)


#f1="../../data/keras_lstm_mnist/abs_model/10_15_benign.log"
#f2="../../data/keras_lstm_mnist/abs_model/10_15_faults.log"
import sys
ast_file,acc_file = sys.argv[1],sys.argv[2]
tl = "" if len(sys.argv) <=3  else sys.argv[3].strip()
f1,f2 = ast_file.strip(),acc_file.strip()



data11=pd.read_csv(f1,index_col=False,names=["benign_truth","benign_truth2","benign_truth_avg_dis"])
data12=pd.read_csv(f2,names=["fault_truth","fault_truth2","fault_truth_avg_dis","fault_pred","fault_pred2","fault_pred_avg_dis"])



data1 = functools.reduce(lambda  left,right: pd.merge(data11,data12,left_index=True,right_index=True,
                                    how='inner'), [data11,data12] )

idx = data1.index.values

fg = plt.figure()


cl = "benign_truth"
plt.plot(idx,data1[cl].values ,label=cl,)
cl = "fault_truth"
plt.plot(idx,data1[cl].values ,label=cl,)
cl = "fault_pred"
plt.plot(idx,data1[cl].values ,label=cl,)

dim=[1, 20, 40, 60, 80,100]
plt.xticks(dim)
plt.ylabel('percentage')
plt.xlabel('top-k')
plt.legend(loc='lower right')
fg.savefig(os.path.join(here_dir,"metric1.pdf"), bbox_inches='tight')


##########
##
##########
fg = plt.figure()

idxx = np.array(idx)+1
cl = "benign_truth"
plt.plot(idx,data1[cl].values/idxx ,label=""+cl)
cl = "fault_truth"
plt.plot(idx,data1[cl].values/idxx ,label=""+cl)
cl = "fault_pred"
plt.plot(idx,data1[cl].values/idxx ,label=""+cl)

plt.ylabel('percentage')
plt.xlabel("top-k")
plt.xticks(dim)

plt.legend(loc='upper right')
fg.savefig(os.path.join(here_dir,"metric2.pdf"), bbox_inches='tight')

##########
##
##########
fg = plt.figure()

cl = "benign_truth_avg_dis"
plt.plot(idx,data1[cl].values ,label=cl.replace("_avg_dis",""))
cl = "fault_truth_avg_dis"
plt.plot(idx,data1[cl].values ,label=cl.replace("_avg_dis",""))
cl = "fault_pred_avg_dis"
plt.plot(idx,data1[cl].values ,label=cl.replace("_avg_dis",""))

plt.ylabel('avg_dis')
plt.xlabel('top-k')
plt.xticks(dim)
plt.legend(loc='upper right')
fg.savefig(os.path.join(here_dir,"metric3.pdf"), bbox_inches='tight')
print ("finish")
