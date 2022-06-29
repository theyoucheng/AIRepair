# COCO experiments

**Record standard outputs and files in each log_dir.**

### Train original model
```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
```


### Repair confusion error(new bn)
```
python2 repair_confusion_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair --first "woman" --second "bowl" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.5
```



### Repair bias error(new bn)
```
python2 repair_bias_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair --first "bowl" --second "woman" --third "man" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.5
```

### Get confusion results
```
python3 get_results.py --original_model 'global_epoch_confusion.npy' --repaired_model 'coco_confusion_repair/global_epoch_confusion.npy' --first "woman" --second "bowl" --confusion
```
Results are saved to *coco_confusion.pdf*  

### Get bias results
```
python3 get_results.py --original_model 'global_epoch_confusion.npy' --repaired_model 'coco_bias_repair/global_epoch_confusion.npy' --first "bowl" --second "woman" --third "man" --bias
```
Results are saved to *coco_bias.pdf*  




