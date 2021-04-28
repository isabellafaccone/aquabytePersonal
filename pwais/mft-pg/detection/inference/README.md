AMI location:

amazon/Deep Learning AMI (Ubuntu 18.04) Version 43.0

when started, free up 39GB of disk :((((
`rm -rf /home/ubuntu/anaconda3/`

using this container: 
https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_20-08.html#rel_20-08

nvcr.io/nvidia/tensorrt:20.08-py3

note!! the cuda is newer than AOS (is 7.1.3)
Ubuntu 18.04
Note: Container image 20.08-py3 contains Python 3.6.
NVIDIA CUDA 11.0.3 including cuBLAS 11.2.0. ****
NVIDIA cuDNN 8.0.2

Build the image for x86:

```
cd mft-pg/detection/inference/docker
docker build -t mft-pg-inference-x86 .
```

Start a container, run TensorRT for your model, test inference:
```
host# nvidia-docker run -it -v ~/mft-pg:/opt/mft-pg -w /opt/mft-pg mft-pg-inference-x86 bash
indocker$ cd /opt/mft-pg/detection/inference/models/

# This may take several minutes:
indocker$ python3 /opt/tensorrt_demos/yolo_to_onnx.py -m yolo4.fish-head
indocker$ python3 /opt/tensorrt_demos/onnx_to_tensorrt.py -m yolo4.fish-head

indocker$ python3 /opt/mft-pg/detection/inference/test_trt.py yolo4.fish-head
```