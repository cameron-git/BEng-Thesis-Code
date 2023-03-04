FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:$PATH"

WORKDIR /app

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y git python3-pip
RUN apt-get install -y gcc-10 g++-10

# AEStream
RUN CC=gcc-10 CXX=g++-10 pip install git+https://github.com/norse/aestream.git@feature-torch