# COCO experiments

**Record standard outputs and files in each log_dir.**

### Train original model
```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/'
```

### Repair confusion error (w-bn(0.5))
```
python2 repair_confusion_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair --first "person" --second "bus" --ann_dir '../../../../coco/annotations' --image_dir '../../../../coco/' --replace --ratio 0.5
```

### Get original model results
```
python get_original_confusion.py --pretrained ../../../models/coco_original_model/model_best.pth.tar --log_dir ../../../models/coco_original_model --first "person" --second "bus" --ann_dir ../../../../coco/annotations --image_dir ../../../../coco
```

### Get confusion results
```
python3 get_results.py --original_model 'global_epoch_confusion.npy' --repaired_model 'coco_confusion_repair/global_epoch_confusion.npy' --first "person" --second "bus" --confusion
```
Results are saved to *coco_confusion.pdf*  
