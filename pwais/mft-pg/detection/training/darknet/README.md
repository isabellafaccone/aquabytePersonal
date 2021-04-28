# Darknet: Training & Generating Detection Fixtures

Note that training is done in a separate environment at this time because:
 1) The training machine (Aquabyte Lambda Quad) has an old version of CUDA
      and we can't change it right now.
 2) We need to convert the model to TensorRT for inference (through ONNX),
      and the target inference environment is a Jetson TX2, so inference
      needs to happen in a different environment / docker container anyways.

## Build the training environment

This command builds a docker image for training locally with a machine
that has CUDA 9 (e.g. the Lambda Quad machines).  We don't bother pushing
the image to ECR because we anticipate we'll throw it away soon.

```
cd $HOME/research-exploration/pwais/mft-pg/detection/training/darknet/docker
docker build -t mft-pg-darknet-train .
```

## Train a model

This command trains a model and saves the weights and debug assets to 
this source tree.

```
cd $HOME/research-exploration/pwais/mft-pg/detection/training/darknet/
nvidia-docker run --rm -it \
  -v`pwd`/models/fish-body-v1:/opt/mft-pg-train  \
  -v`pwd`/../../../datasets:/opt/datasets \
  -w /opt/mft-pg-train \
  -e CUDA_VISIBLE_DEVICES=3 \
    mft-pg-darknet-train train.sh
```

## Generate a fixture of detection bounding boxes

This command runs a model on all images in a dataset and outputs a
detections asset that can be used as input to a tracker.


