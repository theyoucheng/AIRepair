import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import functools

from matplotlib import pyplot as plt, rcParams

import os
import json

px = {"axes.labelsize": 20, "axes.titlesize": "x-large"}
px = {'legend.fontsize': 'x-large',
      #          'figure.figsize': (15, 5),
      'axes.labelsize': 'x-large',
      'axes.titlesize': 'x-large',
      'xtick.labelsize': 'x-large',
      'ytick.labelsize': 'x-large'}

rcParams.update(px)


def normlize_fn(data):
    # demonstrate data normalization with sklearn
    is_sing = False
    if len(data.shape) <= 1:
        data = data.reshape(-1, 1)
        is_sing = True
    scaler = MinMaxScaler()
    # fit scaler on data
    scaler.fit(data)
    # apply transform
    normalized = scaler.transform(data)
    if is_sing:
        normalized = normalized.reshape(-1)
    # inverse transform
    # inverse = scaler.inverse_transform(normalized)
    return normalized


def read_file_normlize(file_path, columns=["ids", "bic", "converage", "invalid_1", "invalid_2", "invalid_3"],
                       skip_rows=None):
    '''
    read csv and  normalize the columns which if has big than 1 data
    '''
    data = pd.read_csv(file_path, names=columns, header=None, skiprows=skip_rows)
    for cl in columns:
        if "ids" == cl:
            continue
        x = data[cl]
        # norm2 = normalize(x[:,np.newaxis], axis=0).ravel()
        if x.values.max() > 1 or x.values.max() < 0:
            norm2 = normlize_fn(x.values)  # normalize(x[:,np.newaxis], axis=0).ravel()
        else:
            norm2 = x.values
        data[cl] = norm2

    return data


def plot_picture(dt_frame_list, subplot_handle=None):
    '''
    plot_index_str : idx_total
    '''
    if subplot_handle is None:
        subplot_handle = plt
    all_ids = [one_frame["ids"] for one_frame in dt_frame_list]
    idx = all_ids[0]

    if len(dt_frame_list) > 1:
        print("merge...")
        dt_frame = functools.reduce(lambda left, right: pd.merge(left, right, on=['ids'],
                                                                 how='inner'), dt_frame_list)
    else:
        dt_frame = dt_frame_list[0]

    dt_frame = dt_frame.loc[dt_frame.ids <= 80]
    idx = dt_frame["ids"].values
    i = 0
    for cl in dt_frame.columns:
        if "invalid" in cl or "ids" == cl:
            continue
        subplot_handle.plot(idx, dt_frame[cl], label=cl)
        i += 1

    subplot_handle.legend(loc='upper right')
    # subplot_handle.title = "fn_name"


import sys

here_dir = os.path.dirname(__file__)

ast_file, acc_file = sys.argv[1], sys.argv[2]
tl = "" if len(sys.argv) <= 3 else sys.argv[3].strip()
ast_file, acc_file = ast_file.strip(), acc_file.strip()
# ast_file = "./10_30_log.ast"
# acc_file = "./10_30_acc.log"

############
# skip_firstrow=None # first row usally abnorm,
skip_firstrow = [0]  # remove first row to avoid  abnorm,

############
for row_indx, skip_firstrow in enumerate([None]):
    axs = plt
    fg = plt.figure()

    #########
    ### combine two log files togather
    #########
    dt_frame_list = []
    fn_name = ast_file  # "./10_30_log.ast"
    columns = ["ids", "BIC", "Avg_Stable", "invalid_1", "invalid_2", "invalid_3"]
    dt_frame_list.append(read_file_normlize(file_path=fn_name, columns=columns, skip_rows=skip_firstrow))

    fn_name = acc_file  # "./10_30_acc.log"
    columns = ["ids", "invalid_1", "SimNN_Train_Acc", "SimNN_Test_Acc"]
    dt_frame_list.append(read_file_normlize(file_path=fn_name, columns=columns, skip_rows=skip_firstrow))

    plot_hd = axs
    plot_picture(dt_frame_list, subplot_handle=plot_hd)

# ast_file,acc_file

get_savefn = lambda x: [os.path.basename(x).split(".")[0]] + [y for y in os.path.dirname(x).split("/") if
                                                              "lstm" in y or "gru" in y]

dim = [1, 20, 40, 60, 80, 100]
plt.xticks(dim)

plt.xlabel("#Components")

plt.legend(loc='upper right')
# plt.title("k=3")
if tl:
    plt.title(tl)
save_fig_path = "_".join(get_savefn(ast_file) + get_savefn(acc_file)) + ".pdf"
save_fig_path = os.path.join(here_dir,save_fig_path)

print(save_fig_path)
fg.savefig(save_fig_path, bbox_inches='tight')
# plt.savefig(save_fig_path)

# plt.show()
# exit()
