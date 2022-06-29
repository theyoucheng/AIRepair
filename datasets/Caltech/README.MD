# Dataset name

Caltech-101

## Description

Caltech-101 consists of pictures of objects belonging to 101 classes, plus one background clutter class. Each image is labelled with a single object. Each class contains roughly 40 to 800 images, totalling around 9k images. Images are of variable sizes, with typical edge lengths of 200-300 pixels. This version contains image-level labels only. The original dataset also contains bounding boxes.

## Homepage

http://www.vision.caltech.edu/Image_Datasets/Caltech101/

## Source code

[tfds.image_classification.Caltech101](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image_classification/caltech.py)

## Feature structure

```
FeaturesDict({
    'image': Image(shape=(None, None, 3), dtype=tf.uint8),
    'image/file_name': Text(shape=(), dtype=tf.string),
    'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=102),
})
```
## Citation

```
@article{FeiFei2004LearningGV,
  title={Learning Generative Visual Models from Few Training Examples: An Incremental Bayesian Approach Tested on 101 Object Categories},
  author={Li Fei-Fei and Rob Fergus and Pietro Perona},
  journal={Computer Vision and Pattern Recognition Workshop},
  year={2004},
}
```

