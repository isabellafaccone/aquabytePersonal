# Convolutional Pose Machines - Tensorflow


This is the **Tensorflow** implementation of [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release), one of the state-of-the-art models for **fish pose estimation**.

## With some additional features:
 - Easy multi-stage graph construction
 - Kalman filters for smooth pose estimation

## Environments
 - Windows 10 / Ubuntu 16.04
 - Tensorflow 1.2.0
 - OpenCV 3.2

## How to start
### Download Data
run the following command in terminal:
python utils/get_dataset_from_keypoints.py

This will download data from the "keypoints.json" file into a training and validation set. Images are downloaded to utils/dataset/training/fish/img/. There is a label file at utils/dataset/training/fish/labels.txt. These will be used to generate TFRecords for training.

### Generate Training Data
run the following command in terminal:
python utils/create_cpm_tfr_fulljoints.py

This will generate a TFRecord used for training from the images and keypoints in utils/dataset/training/fish/. The TFRecord will be saved at "./cpm_sample_dataset_512x512.tfrecords".

### Train a network
run the following command in terminal:
python train.py

This will start training a network from the TFRecord you just generated. Output will be printed in the console, but the training module has also been integrated with tensorboard to allow for visualization during training. The Tensorboard output is saved in the directory named "logs". To view these logs, run the following command:
tensorboard --logdir=logs/

You will be able to see the loss curves and heatmap output for each stage of the network, as well as the original images and groundtruth heatmaps.

During training, a trained network will be periodically generated and saved in the directory "models".

### Run Inference With a Trained Model
Move the most recently generated network from the directory "models" to the directory "trained_models". Then run the following command:
python demo_cpm.py

This use the trained model to generate heatmaps for every image in the directory "utils/dataset/validation/fish". The heatmaps will be converted into x/y coordinates using either the "argmax" or "center of mass" calculation. The prediction results as well as ground truth will be presented on your screen (press enter to continue to the next image). Euclidean and other distance metrics will soon be added and logged to the console for each image and averaged for each image in the validation set.


