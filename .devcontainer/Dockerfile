# .devcontainer/Dockerfile

# Use the existing base image
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ca-certificates \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA package repositories
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.1-515.65.01-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.1-515.65.01-1_amd64.deb && \
    apt-key add /var/cuda-repo-ubuntu2004-11-7-local/7fa2af80.pub && \
    apt-get update && \
    apt-get -y install cuda

# Install cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=8.4.1.50-1+cuda11.7 && \
    rm -rf /var/lib/apt/lists/*

# Ensure pip is up-to-date
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
