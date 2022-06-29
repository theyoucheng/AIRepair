import argparse
import numpy as np
from sklearn.metrics import average_precision_score

# python repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode bias --first skis --second woman --third man

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default='original_test_data.npy ')
parser.add_argument("--eta", type=float, default=0.1)
parser.add_argument("--eta2", type=float, default=1)
parser.add_argument("--mode", type=str, default='confusion')
parser.add_argument("--first", type=str, default='skis')
parser.add_argument("--second", type=str, default='woman')
parser.add_argument("--third", type=str, default='man')



args = parser.parse_args()

id2object = {
 0: 'man',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush',
 80: 'woman'}


m = len(id2object)
object2id = {v:k for k,v in id2object.items()}


np_original = np.load(args.data_file, allow_pickle=True).item()
eta = args.eta
eta2 = args.eta2
mode = args.mode

target_class_1 = object2id[args.first]
target_class_2 = object2id[args.second]
target_class_3 = object2id[args.third]


features = np.concatenate(np_original["features"])

def sigmoid(z):
    return 1/(1 + np.exp(-z))







features = sigmoid(features)
print(features.shape)
# print(features[:2])

preds_object = []

yhats = []

for i in range(features.shape[0]):
    yhat = []
    feature = features[i]
    # confusion
    if mode == 'confusion':
        if feature[target_class_1] > 0.5 and feature[target_class_2] > 0.5:
            feature[target_class_1] *= eta
            feature[target_class_2] *= eta
    elif mode == 'bias':
        if (feature[target_class_1] > 0.5 and feature[target_class_2] > 0.5) or (feature[target_class_1] > 0.5 and feature[target_class_3] > 0.5):
            feature[target_class_1] *= eta
            feature[target_class_2] *= eta
            feature[target_class_3] *= eta

    for j, f in enumerate(feature):
        if f > 0.5:
            yhat.append(id2object[j])
    yhats.append(yhat)
    preds_object.append(feature)

preds_object = np.array(preds_object)

targets_object = np.zeros_like(features)

for i, label_list in enumerate(np_original['labels']):
    target_list = []
    for label in label_list:
        id = object2id[label]
        targets_object[i, id] = 1
eval_score_object = average_precision_score(targets_object, preds_object)



type2confusion = {}
pair_count = {}
confusion_count = {}
labels = np_original['labels']
for li, yi in zip(labels, yhats):
    no_objects = [id2object[i] for i in range(m) if id2object[i] not in li]
    for i in li:
        for j in no_objects:
            if (i, j) in pair_count:
                pair_count[(i, j)] += 1
            else:
                pair_count[(i, j)] = 1

            if i in yi and j in yi:
                if (i, j) in confusion_count:
                    confusion_count[(i, j)] += 1
                else:
                    confusion_count[(i, j)] = 1
object_list = id2object.values()
for i in object_list:
    for j in object_list:
        if i == j or (i, j) not in confusion_count or pair_count[(i, j)] < 10:
            type2confusion[(i, j)] = 0
            continue
        type2confusion[(i, j)] = confusion_count[(i, j)]*1.0 / pair_count[(i, j)]

print('mean average precision:', eval_score_object)
# print('keys',type2confusion.keys())
conf12 = type2confusion[(args.first, args.second)]
conf21 = type2confusion[(args.second, args.first)]
if args.mode == 'confusion':
    print('confusion:', (conf12+conf21)/2, conf12, conf21)
elif args.mode == 'bias':
    conf13 = type2confusion[(args.first, args.third)]
    conf31 = type2confusion[(args.third, args.first)]
    print('confusion12:', (conf12+conf21)/2, conf12, conf21)
    print('confusion13:', (conf13+conf31)/2, conf13, conf31)
    print('bias:', np.abs((conf12+conf21)-(conf13+conf31)) / 2)
