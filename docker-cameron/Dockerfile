FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y git python3-pip
RUN apt-get install -y gcc-10 g++-10

# PyTorch
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# AEStream
RUN CC=gcc-10 CXX=g++-10 pip install git+https://github.com/norse/aestream.git@feature-torch

# Metavision SDK
RUN apt -y install libcanberra-gtk-module mesa-utils ffmpeg
RUN apt -y install python3.8-dev
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install "opencv-python>=4.5.5.64" "sk-video==1.1.10" "fire==0.4.0" "numpy<=1.21" pandas scipy h5py
RUN python3 -m pip install jupyter jupyterlab matplotlib "ipywidgets==7.6.5"
RUN apt -y install cmake libboost-program-options-dev libeigen3-dev
COPY metavision.list /etc/apt/sources.list.d
RUN apt update
RUN apt -y install metavision-sdk

# Norse
RUN pip install norse

# Tonic
RUN pip install tonic