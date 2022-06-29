Model-based influence analysis for RNN
======
This repository contains our model-based analysis framework for the NIPS submission.


## Usage
### Install
```
## cuda==11.1
conda install pytorch torchvision  cudatoolkit=11.1 -c pytorch -c nvidia 
#conda install pytorch torchvision  cudatoolkit=10 -c pytorch  
## cuda==10.1 
conda install -c pytorch=1.8.0 torchtext=0.6.0
pip install  tensorflow_gpu==2.5.0
```

```
pip install -r requirements.txt
python -m spacy download en
pip install -e ./
```
### Download
```
cd data/
sh download_prepare.sh
python preprocess.py
python preprocess_sst.py
```

* All command lines below have been tested on this env

|   installed    | Version |
| ----------- | ----------- |
| pytorch      | 1.8.1       |
| tensorflow   | 2.5.0        |
| torchvision   | 0.9.1        |
| torchtext   | 0.6.0        |
| spacy | 3.1.0 |
|nltk|3.6.2 | 


**Note that the results depend on the precision of the automata. 
In the example, we just randomly set some parameters (e.g., we set the gmm with 7 components for a quick run of the program), 
so the results may not be good due to that the autamoton is not the best one.
To get better results, please select better automaton. For example, in the paper, we select 43, 37 and 22 components for MNIST, TOXIC and IMDb**


### For Section B.1:

RQ1 includes two steps: building the automata and evaluate its accuracy

```
step-1:
python apps/RQ1/build_abst_model.py -pca 10 -epoch 15 -start 1 -end 10 -model keras_lstm_mnist -path save/
```


The meanings of the options are:

1. `-pca` determines the number of dimensions to be reduced by PCA
2. `-epoch` determines the epoch for the target model. If the model does not exist, we will train one.
3. `-start` start number of the components for the GMM
4. `-end` ending number of the components for the GMM.
5. `-model` chooses the target model and dataset from `keras_lstm_mnist`, `torch_gru_imdb`, `torch_gru_toxic`,  `torch_gru_sst`
6. `-path` chooses the output path

After the step 1, we will get one RNN model, several abstract models (i.e., automata) with different numnbers of components and the extracted features (for traning/test data) with the abstract models.
It logs the information of the abstract models in `save/keras_lstm_mnist/abs_model/10_10_s1_abst.log`


```
step-2 evaluate the accuracy:
python apps/RQ1/train_features.py -pca 10 -epoch 15 -start 1 -end 10 -model keras_lstm_mnist -path save
```
`path_dir` is the path in step-1. After the step 2, we will get the SimNN accuracy in `save/keras_lstm_mnist/abs_model/10_10_acc.log`


* plot refinement of PCA configurations（Section A.3.1, i.e., Figure 4 in paper）
```
python  apps/RQ1/pltxyz.py save/keras_lstm_mnist/abs_model/10_15_s1_abst.log  save/keras_lstm_mnist/abs_model/10_15_acc.log  "k=10"
```
There is one pdf file generated in RQ1 directory.

**To reproduce the similar results for RQ1, please set -end with 80, PCA with 3 , 10 for MNIST, 20 for IMDb and 10 for TOXIC.
In the following expeirments, please select the better automaton which has m components (by setting -components m) 
based on the stability values in save/keras_lstm_mnist/abs_model/10_15_s1_abst.log.**

### For Section 4.1:
Section 4.1: it will evaluate the accuracy with different features

```
step-3
python apps/RQ1/feature_comparision.py -pca 10 -epoch 15 -components 7 -model keras_lstm_mnist -path save
```
The meanings of the options are:

1. `-pca` determines the number of dimensions to be reduced by PCA
2. `-epoch` determines the epoch for the target model. If the model does not exist, we will train one.
3. `-components` the number of the components for the GMM generated from RQ1
4. `-model` chooses the target model and dataset from `keras_lstm_mnist`, `torch_gru_imdb`, `torch_gru_toxic`,  `torch_gru_sst`
5. `-path` chooses the output path, which should be the same with the setting in step-1

It will output the results in the stdout.

### For Section A.3.2

Section A.3.2: it will show the pattern for interpreting the faults with two steps.


```
step-1: calculate the metrics for failed inputs
python apps/RQ1/fault_loc.py  -pca 10 -epoch 15 -components 7 -benign 0 -model keras_lstm_mnist -path save
```


```
step-2: calculate the metrics for benign inputs
python apps/RQ1/fault_loc.py  -pca 10 -epoch 15 -components 7 -benign 1 -model keras_lstm_mnist -path save
```

The meanings of the options are:

1. `-pca` determines the number of dimensions to be reduced by PCA
2. `-epoch` determines the epoch for the target model. If the model does not exist, we will train one.
3. `-components` the number of the components for the GMM generated from RQ1
4. `-benign` chooses the data which is benign or failed, 0 for failed, 1 for benign
5. `-model` chooses the target model and dataset from `keras_lstm_mnist`, `torch_gru_imdb`, `torch_gru_toxic`,  `torch_gru_sst`
6. `-path` chooses the output path, which should be the same with the setting in step-1

It outputs the information in `save/keras_lstm_mnist/abs_model/10_10_faults.log` and `save/keras_lstm_mnist/abs_model/10_10_benign.log`  

* plot statistical results for understanding the benign/failed predictions 
```
python apps/RQ1/pltxyz3.py save/keras_lstm_mnist/abs_model/10_15_benign.log  save/keras_lstm_mnist/abs_model/10_15_faults.log
```
Three pdf files are generated i.e., metric1.pdf metric2.pdf and metric3.pdf.

Note that `path_dir` should be the same path with the previous setting.



### For Section 4.2

In order to run experiments for Section 4.2, you need to finish following five steps.  

The all commands are integrated in **./app/RQ3/command.sh**. Running command:
```
cd app/RQ3
bash .command.sh
```
The command.sh file contains the following steps:

(1)  **generate flip data and corresponding models**

(2) **train sgd models** 

(3)  **calculate influence scores with sgd and icml methods**  

(4)  **retrain** 

(5)  **plot results** 



### For Section 4.3
Section 4.3: it will generates data for repairing the common failed inputs (from multiple models) with retraining.


```
step-1: generates data 
# in app/RQ4
python apps/RQ4/datagen.py  -pca 10  -epoch 15 -components 7 -model keras_lstm_mnist -path save
```

The meanings of the options are:

1. `-pca` determines the number of dimensions to be reduced by PCA
2. `-model` chooses the target model and dataset from `keras_lstm_mnist`, `keras_gru_mnist`, `torch_gru_imdb`, `torch_lstm_imdb`. 
3. `-components` the number of the components for the GMM which has the best performance
4. `-path` chooses the output path


The parameters of `k` and model epoches `pca_models` can be changed in the source file.
It will generates the data in `save/keras_lstm_mnist/temp_files/`.

**To reproduce the results for RQ4, you can train multiple models (e.g., 5,8,10,12,15,18) to obtain stable faults
and generate the best automata for these models.
However, it will spend much time. 
We also included the generated data in the paper (see `apps/RQ4/retrain.npz`), so you can just retrain the model
with the following commands that take this npz file as input.**


```
step-2:  retrain the model by adding the generated results

python apps/RQ4/repair_mnist_rnn_keras.py -epoch 15 -type 0 -p  ./apps/RQ4/retrain.npz -start 5 -rnn_type lstm -seed 5
```

```
step-3:  get the results with  the original data

python apps/RQ4/repair_mnist_rnn_keras.py -epoch 15 -type 1 -p  ./apps/RQ4/retrain.npz -start 5 -rnn_type lstm -seed 5
```


```
step-4:  get the results with  the random strategy

python apps/RQ4/repair_mnist_rnn_keras.py -epoch 15 -type 2 -p  ./apps/RQ4/retrain.npz -start 5 -rnn_type lstm -seed 5
```

It will output the results in the stdout.

The meanings of the options are:

1. `-epoch`  determines the epoch for the target model. If the model does not exist, we will train one.
2. `-type` determines whether we add the generated data, 0 for using the new data and 1 for using the original data
3. `-p` the path of the new generated data
4. `-start` the starting epoch for the retraining.
5. `-rnn_type` chooses the target model type from `lstm`, `gru`.
6. `-seed` determines the repeat times. 


### For Section 4.4
Section 4.4: it will repair the postive-to-negative errors by inserting  segments into influential training samples.


```
python apps/RQ4/segment_repair.py  -pca 10  -components 37 -epoch 15  -model torch_gru_toxic -path save

```

The meanings of the options are:

1. `-pca` determines the number of dimensions to be reduced by PCA
2. `-model` chooses the target model and dataset from `keras_lstm_mnist`, `keras_gru_mnist`, `torch_gru_imdb`, `torch_lstm_imdb`. 
3. `-components` the number of the components for the GMM which has the best performance
4. `-path` chooses the output path




### For Section B.5

Section B.5: it will show the backdoor attack and fix results with segment level influence analysis


```
python apps/RQ2/infl_backdoor_analysis.py  -pca 10 -epoch 40 -components 37 -round 1 -path save
```


It will output the results including backdoor attack rate and fix rate in the stdout.
The meanings of the options are:

1. `-pca` determines the number of dimensions to be reduced by PCA
2. `-epoch` determines the epoch for the target model. If the model does not exist, we will train one.
3. `-components` the number of the components for the GMM generated from RQ1
4. `-round` chooses how many seeds we want to run
5. `-path` chooses the output path, which should be the same with the setting in step-1
