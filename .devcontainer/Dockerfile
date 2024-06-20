# .devcontainer/Dockerfile

# Use the latest CUDA runtime image
FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

# Set environment variable to noninteractive to avoid geographic area prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set CUDA environment variables
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    swig \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    ca-certificates \
    gnupg \
    software-properties-common \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set the time zone non-interactively
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

# Install CUDA toolkit
RUN apt-get update && apt-get install -y \
    cuda-toolkit-12-5

# Install PyTorch with CUDA 12.5 support
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu125

# Ensure pip is up-to-date and install setuptools
RUN pip3 install --upgrade pip setuptools

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
