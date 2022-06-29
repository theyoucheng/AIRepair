import pytreebank
import sys
import os

import tqdm 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

import numpy as np 

remove_list = stopwords.words('english') + list(string.punctuation)

raise Exception("path problem")
dataset = pytreebank.load_sst('/home/malei/.torch/.data/sst/trees')

root =os.path.expanduser("./sst_data/")

def process_xtrain_ytrain(all_items,filter_remove_neutral=True):
    x_list = []
    y_list = []
    
    
    for item in tqdm.tqdm(all_items):
        label = item.to_labeled_lines()[0][0]
        if filter_remove_neutral and label==2:
            continue
        
        label  =0 if  label <2 else 1 
        x_list.append(item.to_labeled_lines()[0][1])
        y_list.append(label)
    
    return x_list, y_list 

def process_tuple(all_items,filter_remove_neutral=True):
    x_list = []
    
    for item in tqdm.tqdm(all_items):
        text= item.to_labeled_lines()[0][1]
        
        label=item.to_labeled_lines()[0][0]
        if filter_remove_neutral and label==2:
            continue
        label  ="neg" if  label <2 else "pos" 

        # text_processed_list = [word for word in word_tokenize(text.lower()) if word not in remove_list]
        text_processed_list = [word for word in word_tokenize(text.lower()) ]

        x_list.append((label,text_processed_list))
    return x_list 


os.makedirs(os.path.join(root,"data"), exist_ok=True)
os.makedirs(os.path.join(root,"data_list"), exist_ok=True)

import joblib 
# Store train, dev and test in separate files
import pandas  as pd 

for category in ['train', 'test', 'dev']:
    print ("prcess...",category)
    d1,d2= process_xtrain_ytrain( dataset[category] )
    d3 = process_tuple(dataset[category] )

    npz=os.path.join(root,"data",f"sst_{category}.npz") 
    np.savez(npz,train=d1,label=d2)
    
    npl=os.path.join(root,"data_list",f"sst_{category}.lst") 
    joblib.dump(d3,npl)
    
    np_csv=os.path.join(root,"data_list",f"sst_{category}.csv")
    
    df = pd.DataFrame(d3, columns=["label","text"])
    df.to_csv(np_csv, index=False)
