# spermatozoids

Cells detection and classification(regression) on microscopic images.

Project demonstrates application of convolutional neural networks and deep learning for biological data analysis. Neural network implemented on [Mat??](https://github.com/victorlempitsky/Mate).

## How to use it

This framework is working in Matlab.

1. First, install [MatConvNet](http://www.vlfeat.org/matconvnet/)

2. Install [Mat??](https://github.com/victorlempitsky/Mate)

3. Clone this repo

4. >>> setup

5. You are good to go! Try pretrained models using "apply(_path_to_your_image_)", or label your data using "label_centers.m" and "label_orientations.m" and then train model defined in "init_..." by running "train_detection.m"