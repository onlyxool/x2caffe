FROM ubuntu:22.04

LABEL maintainer=onlyxool@gmail.com

ARG LC_ALL="C"
ARG DEBIAN_FRONTEND="noninteractive"
ARG TZ="Etc/UTC"
RUN ln -snf /usr/share/zoneinfo/"${TZ}" /etc/localtime && echo "${TZ}" >/etc/timezone


RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      wget \
      libatlas-base-dev \
      libboost-all-dev \
      libgflags-dev \
      libgoogle-glog-dev \
      libhdf5-serial-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libprotobuf-dev \
      libsnappy-dev \
      protobuf-compiler \
      python3-dev \
      python3-numpy \
      python3-pip \
      python3-setuptools \
      python3-scipy && \
    rm -rf /var/lib/apt/lists/*


ARG CAFFE_VERSION="v1.0.1"
ARG CAFFE_SOURCE_DIR="/caffe"
ARG CAFFE_BUILD_DIR="/caffe/build"
ARG CAFFE_INSTALL_PREFIX="/workspace/x2caffe/libcaffe"

RUN for req in $(cat "${CAFFE_SOURCE_DIR}"/python/requirements.txt) pydot; do pip3 install --upgrade --no-cache-dir "${req}"; done 
#RUN pip3 install Cython>=0.19.2 numpy>=1.7.1 scipy>=0.13.2 scikit-image>=0.9.3 matplotlib>=1.3.1 ipython>=3.0.0 h5py>=2.2.0 leveldb>=0.191 networkx>=1.8.1 nose>=1.3.0 pandas>=0.12.0 python-dateutil>=1.4 protobuf>=2.5.0,<4 python-gflags>=2.0 pyyaml>=3.10 Pillow>=2.3.0 six>=1.1.0
RUN pip3 install --no-cache-dir onnx==1.14.1  onnxruntime==1.15.1 onnxoptimizer==0.3.13 onnxsim 
RUN pip3 install streamlit
EXPOSE 8501

ENV PYTHONPATH="${CAFFE_INSTALL_PREFIX}/python:${PYTHONPATH}"
ENV PATH="${CAFFE_INSTALL_PREFIX}/bin:${CAFFE_INSTALL_PREFIX}/python:${PATH}"



ENV THIS_DOCKER_REPOSITORY="https://hub.docker.com/r/onlyxool/streamlit"
ENV THIS_DOCKER_TAG="cpu"
ARG BUILD_DATE
ENV THIS_DOCKER_CREATED="${BUILD_DATE}"
ENV THIS_GIT_REPOSITORY="https://github.com/duruyao/caffe"
ENV THIS_GIT_TAG="v1.0.1"


WORKDIR /workspace