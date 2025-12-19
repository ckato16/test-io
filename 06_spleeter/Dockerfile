FROM python:3.8-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed by Spleeter and h5py
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Spleeter
RUN pip install --no-cache-dir spleeter==2.4.2

# Create working directory
WORKDIR /data

# Start container with bash
ENTRYPOINT ["/bin/bash"]
