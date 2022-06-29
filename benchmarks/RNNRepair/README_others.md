collection of all running cases
======
This file contains our all submission.
### For Section B.1:

```
step-1:

python apps/RQ1/build_abst_model.py -pca 10 -epoch 15 -start 1 -end 10 -model keras_lstm_mnist -path save

python apps/RQ1/build_abst_model.py -pca 10 -epoch 15 -start 1 -end 10 -model torch_lstm_mnist -path save

python apps/RQ1/build_abst_model.py -pca 10 -epoch 15 -start 1 -end 10 -model torch_gru_toxic -path save

python apps/RQ1/build_abst_model.py -pca 10 -epoch 15 -start 1 -end 10 -model torch_gru_sst -path save
```




```
step-2 evaluate the accuracy:

python apps/RQ1/train_features.py -pca 10 -epoch 15 -start 1 -end 10 -model keras_lstm_mnist -path save

python apps/RQ1/train_features.py -pca 10 -epoch 15 -start 1 -end 10 -model torch_lstm_mnist -path save

python apps/RQ1/train_features.py -pca 10 -epoch 15 -start 1 -end 10 -model torch_gru_toxic -path save

python apps/RQ1/train_features.py -pca 10 -epoch 15 -start 1 -end 10 -model torch_gru_sst -path save

```





* plot refinement of PCA configurations（Section A.3.1, i.e., Figure 4 in paper）
```
python  apps/RQ1/pltxyz.py save/keras_lstm_mnist/abs_model/10_15_s1_abst.log  save/keras_lstm_mnist/abs_model/10_15_acc.log  "k=10"

python  apps/RQ1/pltxyz.py save/torch_lstm_mnist/abs_model/10_15_s1_abst.log  save/torch_lstm_mnist/abs_model/10_15_acc.log  "k=10"

python  apps/RQ1/pltxyz.py save/torch_gru_toxic/abs_model/10_15_s1_abst.log  save/torch_gru_toxic/abs_model/10_15_acc.log  "k=10"

python  apps/RQ1/pltxyz.py save/torch_gru_sst/abs_model/10_15_s1_abst.log  save/torch_gru_sst/abs_model/10_15_acc.log  "k=10"
```


### For Section 4.1:
```
step-3

python apps/RQ1/feature_comparision.py -pca 10 -epoch 15 -components 7 -model keras_lstm_mnist -path save

python apps/RQ1/feature_comparision.py -pca 10 -epoch 15 -components 7 -model torch_lstm_mnist -path save

python apps/RQ1/feature_comparision.py -pca 10 -epoch 15 -components 7 -model torch_gru_toxic -path save

python apps/RQ1/feature_comparision.py -pca 10 -epoch 15 -components 7 -model torch_gru_sst -path save
```





### For Section A.3.2
```
step-1: calculate the metrics for failed inputs
python apps/RQ1/fault_loc.py  -pca 10 -epoch 15 -components 7 -benign 0 -model keras_lstm_mnist -path save

python apps/RQ1/fault_loc.py  -pca 10 -epoch 15 -components 7 -benign 0 -model torch_lstm_mnist -path save

python apps/RQ1/fault_loc.py  -pca 10 -epoch 15 -components 7 -benign 0 -model torch_gru_toxic -path save

python apps/RQ1/fault_loc.py  -pca 10 -epoch 15 -components 7 -benign 0 -model torch_gru_sst -path save
```

```
step-2: calculate the metrics for benign inputs

python apps/RQ1/fault_loc.py  -pca 10 -epoch 15 -components 7 -benign 1 -model keras_lstm_mnist -path save

python apps/RQ1/fault_loc.py  -pca 10 -epoch 15 -components 7 -benign 1 -model torch_lstm_mnist -path save

python apps/RQ1/fault_loc.py  -pca 10 -epoch 15 -components 7 -benign 1 -model torch_gru_toxic -path save

python apps/RQ1/fault_loc.py  -pca 10 -epoch 15 -components 7 -benign 1 -model torch_gru_sst -path save
```





* plot statistical results for understanding the benign/failed predictions 
```
python apps/RQ1/pltxyz3.py save/keras_lstm_mnist/abs_model/10_15_benign.log  save/keras_lstm_mnist/abs_model/10_15_faults.log

python apps/RQ1/pltxyz3.py save/torch_lstm_mnist/abs_model/10_15_benign.log  save/torch_lstm_mnist/abs_model/10_15_faults.log

python apps/RQ1/pltxyz3.py save/torch_gru_toxic/abs_model/10_15_benign.log  save/torch_gru_toxic/abs_model/10_15_faults.log

python apps/RQ1/pltxyz3.py save/torch_gru_sst/abs_model/10_15_benign.log  save/torch_gru_sst/abs_model/10_15_faults.log
```


### For Section 4.2
```
cd app/RQ3
bash .command.sh
```



### For Section 4.3
```
python apps/RQ4/datagen.py  -pca 10  -epoch 15 -components 7 -model keras_lstm_mnist -path save
####python apps/RQ4/datagen.py  -pca 10  -components 7 -model torch_lstm_mnist -path save
####python apps/RQ4/datagen.py  -pca 10  -components 7 -model torch_gru_toxic -path save
####python apps/RQ4/datagen.py  -pca 10  -components 7 -model torch_gru_sst -path save
```



### For Section 4.4
```
python apps/RQ4/segment_repair.py  -pca 10  -components 37 -epoch 15  -model torch_gru_toxic -path save
```














