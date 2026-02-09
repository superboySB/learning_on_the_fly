# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu20.04

LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

ENV DEBIAN_FRONTEND=noninteractive

# Default proxy: one-step build in common local setup.
ENV http_proxy=http://127.0.0.1:8889
ENV https_proxy=http://127.0.0.1:8889
ENV HTTP_PROXY=http://127.0.0.1:8889
ENV HTTPS_PROXY=http://127.0.0.1:8889
ENV no_proxy=localhost,127.0.0.1,::1
ENV NO_PROXY=localhost,127.0.0.1,::1

SHELL ["/bin/bash", "-lc"]

# Hard requirements: Ubuntu 20.04 + CUDA 12.9.
RUN . /etc/os-release && test "${VERSION_ID}" = "20.04"
RUN nvcc --version | grep -E "release 12\\.9"

# Base tools + build dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    software-properties-common \
    git \
    tmux \
    vim \
    gedit \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libpng-dev \
    libjpeg-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    build-essential \
    cmake \
    pkg-config \
    bzip2 \
 && rm -rf /var/lib/apt/lists/*

# Install Python 3.9 as default python3 (no conda/micromamba).
RUN add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3.9-venv \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 \
 && update-alternatives --set python3 /usr/bin/python3.9 \
 && ln -sf /usr/local/bin/pip /usr/local/bin/pip3 \
 && python3 --version | grep -E "3\\.9\\."

# Install ROS Noetic (ROS1).
RUN mkdir -p /etc/apt/keyrings \
 && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    | gpg --dearmor -o /etc/apt/keyrings/ros-archive-keyring.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" \
    > /etc/apt/sources.list.d/ros1.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-rospy \
    ros-noetic-geometry-msgs \
    ros-noetic-nav-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    ros-noetic-std-srvs \
    ros-noetic-tf2-ros \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-vcstool \
    python3-catkin-tools \
    python3-rospkg \
    python3-catkin-pkg \
 && rosdep init || true \
 && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=noetic

# Python deps from environment.yml pip section (JAX handled separately).
COPY docker/requirements-pip-core.txt /tmp/requirements-pip-core.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir --ignore-installed -r /tmp/requirements-pip-core.txt \
 && python3 -m pip install --no-cache-dir --ignore-installed \
    catkin-pkg \
    catkin-tools \
    colcon-common-extensions \
    empy \
    netifaces \
    rospkg

# User-requested JAX CUDA installation command.
ARG JAX_CUDA_CUDNN=cuda12_pip
ARG JAX_VERSION=0.4.30
RUN python3 -m pip install --no-cache-dir "jax[${JAX_CUDA_CUDNN}]==${JAX_VERSION}" \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /workspace/learning_on_the_fly
COPY . /workspace/learning_on_the_fly
RUN python3 -m pip install --no-cache-dir -e .

# Runtime defaults.
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc \
 && echo "export ROS_DISTRO=noetic" >> /root/.bashrc \
 && echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH}" >> /root/.bashrc

# Validate core requirements in-image.
RUN . /etc/os-release && test "${VERSION_ID}" = "20.04" \
 && nvcc --version | grep -E "release 12\\.9" \
 && python3 --version | grep -E "3\\.9\\." \
 && bash -lc "source /opt/ros/noetic/setup.bash && python3 -c 'import rospy; print(\"rospy ok\")'" \
 && python3 -c "import jax; print('jax', jax.__version__)"

CMD ["/bin/bash"]
