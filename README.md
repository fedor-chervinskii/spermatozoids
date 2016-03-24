# spermatozoids

Cells detection and classification(regression) on microscopic images.

Project demonstrates application of convolutional neural networks and deep learning for biological data analysis. Neural network implemented on [Mate](https://github.com/victorlempitsky/Mate).

## How to use it

This framework is working in Matlab.

1. First, install [MatConvNet](http://www.vlfeat.org/matconvnet/)

2. Install [Mate](https://github.com/victorlempitsky/Mate)

3. Clone this repo

4. Run *setup.m*

5. You are good to go! Try pretrained models using *apply(_path_to_your_image_)*, or label your data using *label_centers.m* and *label_orientations.m* and then train model defined in *init_...* by running *train_detection.m*

## Research pipeline

If you need to adjust the framework for your own data and tasks, follow the steps:

1. Detection network is trained on patches - collect your patches using your modification of *label_centers.m* or *label_orientatioins.m*
2. After you created banch of files with labels, you want to create database of examples, possibly with augmentation of samples. *collect_dataset_imdb.m* will help you with this. Currently it can rotate (if label is angle it changes accordingly) and jitter.
3. To train network, first create an initialization file like one of *init_xxx_net.m* where set up the network according to your patches' size and desired output and loss (it can be either classification or regression, network could be fully convolutional like *init_det_net.m* means it can accept input of any size and produce matrix of outputs).
4. Training is performed using one of *train_xxx.m* scripts, play with *lr* parameter and number of epochs.
5. To validate the performance and search in the space of hyperparameters on some test images there are couple of drafts for this *validation.m* and *validate_detection.m*
 