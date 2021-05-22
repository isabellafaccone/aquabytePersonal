FROM nvcr.io/nvidia/tensorrt:20.08-py3

# Install Nvidia's extras to get ONNX, numpy, etc
RUN \
  /opt/tensorrt/install_opensource.sh && /opt/tensorrt/python/python_setup.sh

ENV DEBIAN_FRONTEND noninteractive
RUN \
  apt-get update && apt-get install -y \
    sudo wget curl vim git \
    python3-pip python3-opencv
RUN python3 -m pip install --upgrade pip

RUN pip3 install \
      bokeh \
      imageio \
      pandas \
      pycuda \
      ipython \
      imageio-ffmpeg \
      click \
      jupyter \
      mlflow==1.14.1

# Thanks alot click
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Install TRT demo stuff for yolo
RUN \
  mkdir -p /opt && cd /opt && \
  git clone -v https://github.com/jkjung-avt/tensorrt_demos.git/ && \
  cd tensorrt_demos && \
  git checkout 0016973315d1f3f6eaed70a5abd03d6309fe4730
RUN \
  cd /opt/tensorrt_demos && \
  cd ssd && ./install_pycuda.sh && cd - && \
  pip3 install onnx==1.4.1 && \
  cd plugins && make && cd -
ENV PYTHONPATH $PYTHONPATH:/opt/tensorrt_demos

RUN \
  echo "need to edit tensorrt_demos have /opt/tensorrt_demos/plugins/libyolo_layer.so instead of ./plugins/libyolo_layer.so :(" && \
  sed -i 's/\.\/plugins\/libyolo_layer\.so/\/opt\/tensorrt_demos\/plugins\/libyolo_layer\.so/g' /opt/tensorrt_demos/utils/yolo_with_plugins.py && \
  sed -i 's/\.\.\/plugins\/libyolo_layer\.so/\/opt\/tensorrt_demos\/plugins\/libyolo_layer\.so/g' /opt/tensorrt_demos/yolo/plugins.py

# Install tracker stuff
RUN cd /opt/ && \
  git clone https://github.com/adipandas/multi-object-tracker  && \
    cd multi-object-tracker && \
      git checkout 501605262a120f1ab47658f423fa180a6f36117c && \
        pip3 install -r requirements.txt && \
        pip3 install ipyfilechooser && \
        pip3 install -e .
ENV PYTHONPATH $PYTHONPATH:/opt/multi-object-tracker

# Install oarphpy for reports
RUN \
  DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-11-jdk && \
  echo JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 >> /etc/environment && \
  pip3 install pyspark==3.0.1 && \
  pip3 install numpy pandas>=1.0.0
ENV PYSPARK_PYTHON python3
ENV PYSPARK_DRIVER_PYTHON python3
RUN cd /opt/ && \
  git clone https://github.com/pwais/oarphpy.git && \
  cd oarphpy && git checkout v0.1.0 && \
  pip3 install -e .[spark]
ENV PYTHONPATH $PYTHONPATH:/opt/oarphpy