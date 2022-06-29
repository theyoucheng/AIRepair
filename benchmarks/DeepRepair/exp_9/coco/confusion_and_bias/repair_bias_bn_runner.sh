#!/bin/bash
echo "Train original model"
python2 train_epoch_graph.py --log_dir original_model --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
echo "Repair bias with new batch norm"
python2 repair_bias_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair --first "person" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.5
echo "Get the results"
python3 get_results.py --original_model 'global_epoch_confusion.npy' --repaired_model 'coco_bias_repair/global_epoch_confusion.npy' --first "person" --second "clock" --third "bus" --bias


