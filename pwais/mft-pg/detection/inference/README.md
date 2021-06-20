# Inference Experiment Module


## Running on a Jetson TX2

### Flashing your TX2

First you will need:
 * A Jetson TX2 (e.g. a dev kit)
 * A host computer running Ubuntu.  (Some say Virtualbox on a Mac works, but
     it did not work for me and it was extremely slow).
 * An HDMI monitor and keyboard and mouse.  You'll need this in order to accept
     the license once the TX2 boots.

First, you'll need to put your TX2 into recovery mode.  For a video on
what buttons to press and how do this, check out: https://youtu.be/D7lkth34rgM?t=362

I recommend this guide for flashing the TX2 over the terminal on your host
machine:
https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3242/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fquick_start.html%23wwpID0E0XD0HA

At a high level, the process goes like this:
 * On your host: `sudo apt-get install qemu-user-static`
 * Download some NVidia Linux For Tegra (L4T) files from the Nvidia website.
     Use the links in the doc above and be careful to select the correct
     version that you want.  At the time of writing, we're targeting
     JetPack 4.4 and its assocated L4T / TensorRT dependencies.
 * Extract and prepare the flashing files.
 * Run the `flash.sh` script to flash the device.

Once the TX2 is flashed, you need to connect an HDMI monitor and mouse and 
keyboard into order to accept their license and create a user account on
the TX2. (This process is similar to typical set-up for a fresh `Ubuntu`
install).  Once this is done, `ssh` will be available on the TX2 for remote
access over ethernet.

### Fixing `nvidia-docker`

For this demo, we flashed a TX2 with:
  * `Tegra186_Linux_R32.4.3_aarch64.tbz2`
  * `Tegra_Linux_Sample-Root-Filesystem_R32.4.3_aarch64.tbz2`

Sadly, the `nvidia-docker` install is broken in this distro.  For more
information, see:
https://forums.developer.nvidia.com/t/docker-container-nvidia-l4t-ml-wont-run/146534/11

To fix, do this on the TX2:
```
$ sudo apt-get remove nvidia-container-runtime
$ sudo apt-get autoremove
$ reboot

$ sudo apt-get install nvidia-container-runtime
```

Now you should be able to run this on the TX2:
```
docker run --net=host --runtime nvidia  --rm -it nvcr.io/nvidia/l4t-ml:r32.4-py3 bash
```

### Fixing other Nvidia CUDA Dependencies

These packages are required for CUDA in containers to work correctly
but they appear to be missing in the flash install.  Run the following on the
TX2:
```
sudo apt-get install \
  nvidia-container-csv-cuda \
  nvidia-container-csv-cudnn \
  nvidia-container-csv-tensorrt \
  nvidia-container-csv-visionworks \
  cuda-curand-dev-10-2 \
  cuda-toolkit-10-2 \
  cuda-tools-10-2 \
  python3-libnvinfer-dev \
  python3-opencv
```

Test that you have some needed CUDA libs on the host:
```
$ find /usr | grep libnvToolsExt.so
/usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvToolsExt.so
$ find /usr | grep libcurand.so
/usr/local/cuda-10.2/targets/aarch64-linux/lib/libcurand.so
```

And in a container:
```
docker run --runtime=nvidia --rm -it nvcr.io/nvidia/l4t-ml:r32.4.3-py3 bash
root@9ae13c59927d:/# find /usr | grep libnvToolsExt.so
/usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvToolsExt.so
```

### Enabling All the CPUs

You might want to turn on the heatsink fan:
```
sudo echo 255 > /sys/devices/pwm-fan/target_pwm
```

By default it seems the TX2 boots in low power mode and two of the CPUs will
be absent when you run `top`.  Do this to put the TX2 in high power mode and
the CPUs will become available:
```
sudo nvpmodel -m 2
sudo /usr/bin/jetson_clocks --show
```

### Recommended: jtop for Jetson

Install `jtop` and python3 on the host:

```
sudo apt-get install -y python3-dev python3-pip
sudo -H pip3 install -U jetson-stats
sudo systemctl restart jetson_stats.service
```
(You'll need to `sudo reboot`).

### Optional: NFS-mount host code to TX2

Since the TX2 only has 32GBytes of built-in disk, it can make development
easier to mount the your code and artifacts on your x86 dev machine
to the TX2.  This guide works (at the time of writing) for setting up
an NFS export on an x86 host and consuming that export as a TX2 client:
https://www.digitalocean.com/community/tutorials/how-to-set-up-an-nfs-mount-on-ubuntu-20-04






## Running on a Jetson Xavier NX

### Flashing your TX2

First you will need:
 * A Jetson Xavier NX (e.g. a dev kit)
 * A host computer, could be running Mac OS X if you want to use the defaul NVidia boot image.  
    If you want to build an image (e.g. [OE4T Tegra Linux](https://github.com/OE4T/tegra-demo-distro) )
    then you need docker.  See notes on OE4T building below.
 * An HDMI monitor and keyboard (and a mouse would help too, but I was able to set up without one).  
    You'll need this in order to accept the license once the Xavier NX boots.  You won't get
    SSH access to the machine until after a couple reboots of Ubuntu.

### Flash Boot SD Card

Grab the L4T disk image for Xavier NX with Jetpack 4.4 here:
 * https://developer.nvidia.com/jetson-nx-developer-kit-sd-card-image-44

And then flash the card using Balena Etcher:
 * https://www.balena.io/etcher/

Once done, plug the SD card into the Xavier NX board, connect the rest of the stuff and power on.

You will now go through the normal Ubuntu set-up experience and (importantly) need to accept the 
Nvidia license agreement.  You'll need to reboot a couple of times before SSH becomes available.

At the time of writing, NVidia offered the disk image for version 4.5.1, 2021/02/24.
That gave us JetPack 4.5.1:
```
aq@aqxnx:~$ sudo apt-cache show nvidia-jetpack
Package: nvidia-jetpack
Version: 4.5.1-b17
...
```

### Testing Your Xavier NX

Once you have a user account and ssh set-up, `nvidia-docker` should be readily available without
you having to do anything.

Test that you have some needed CUDA libs on the host:
```
$ find /usr | grep libnvToolsExt.so
/usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvToolsExt.so
$ find /usr | grep libcurand.so
/usr/local/cuda-10.2/targets/aarch64-linux/lib/libcurand.so
```

And in a container:
```
docker run --runtime=nvidia --rm -it nvcr.io/nvidia/l4t-ml:r32.4.3-py3 bash
root@9ae13c59927d:/# find /usr | grep libnvToolsExt.so
/usr/local/cuda-10.2/targets/aarch64-linux/lib/libnvToolsExt.so
```

### Recommended: jtop for Jetson

Install `jtop` and python3 on the host:

```
sudo apt-get install -y python3-dev python3-pip
sudo -H pip3 install -U jetson-stats
sudo systemctl restart jetson_stats.service
```
(You'll need to `sudo reboot`).

### Building Boot Image with OE4T

This section is incomplete, but should be able to build an OE4T-based disk image (?).  Edge
team prefers OE4T because it's easier to add Aquabyte-specific stuff into the image.  The
notes below build a vanilla L4T image.

Use this `Dockerfile` to create a builder environment:
```
FROM ubuntu:18.04

RUN \
 apt-get update && \
 DEBIAN_FRONTEND=noninteractive apt-get install -y \
   gawk wget git-core diffstat unzip texinfo gcc-multilib \
   build-essential chrpath socat cpio python3 python3-pip python3-pexpect \
   xz-utils debianutils iputils-ping python3-git python3-jinja2 libegl1-mesa libsdl1.2-dev \
                  pylint3 xterm \
    sudo g++-8 \
    locales \
    git

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
        update-locale LANG=en_US.UTF-8

RUN \
  useradd --password build -ms /bin/bash build && \
  usermod -aG sudo build && \
  echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER build

WORKDIR /home/build/
RUN git clone https://github.com/OE4T/tegra-demo-distro.git
WORKDIR /home/build/tegra-demo-distro
RUN \
  git checkout dunfell-l4t-r32.4.3 && \
  git submodule update --init

ENV LANG en_US.UTF-8      
```

Build using:
`docker build -t pwbuildtest .`

Rung using:
`docker run --rm -it  -v/:/outer_root pwbuildtest bash`

Now build [according to the instructions](https://github.com/OE4T/tegra-demo-distro):
```
build@4dfec0e7d2a9:~/tegra-demo-distro$ . ./setup-env --machine jetson-xavier-nx-devkit
build@4dfec0e7d2a9:~/tegra-demo-distro/build$ bitbake demo-image-full
```



## pwais scratchpad


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
host# nvidia-docker run -it --net=host -v ~/mft-pg:/opt/mft-pg -w /opt/mft-pg mft-pg-inference-x86 bash
indocker$ cd /opt/mft-pg/detection/inference/models/

# This may take several minutes:
indocker$ python3 /opt/tensorrt_demos/yolo/yolo_to_onnx.py -m yolov3-416 -c 2
indocker$ python3 /opt/tensorrt_demos/yolo/onnx_to_tensorrt.py -v -m yolov3-416 -c 2

indocker$ python3 /opt/mft-pg/detection/inference/test_trt.py yolo4.fish-head
```





