
# doh !!! this one is missing tensorRT headers:
FROM nvcr.io/nvidia/l4t-ml:r32.5.0-py3

ENV DEBIAN_FRONTEND noninteractive
RUN \
  apt-get update && apt-get install -y \
    sudo wget curl vim git 
RUN python3 -m pip install --upgrade pip

RUN \
  mkdir -p /opt && cd /opt && \
  git clone https://github.com/jkjung-avt/tensorrt_demos && \
  cd tensorrt_demos && git checkout 0016973315d1f3f6eaed70a5abd03d6309fe4730
RUN \
  cd /opt/tensorrt_demos && \
  cd ssd && ./install_pycuda.sh && cd - && \
  pip3 install onnx==1.4.1 && \
  cd plugins && make && cd -
ENV PYTHONPATH $PYTHONPATH:/opt/tensorrt_demos

