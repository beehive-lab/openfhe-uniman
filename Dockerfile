# Use the base NVIDIA CUDA image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install necessary dependencies including wget and curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    vim \
    ca-certificates \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install CMake 3.25 or higher
RUN curl -L https://cmake.org/files/v3.25/cmake-3.25.0-linux-x86_64.tar.gz | \
    tar --strip-components=1 -xz -C /usr/local && \
    cmake --version

# Clone the repository and switch to the desired branch
RUN git clone https://github.com/beehive-lab/openfhe-uniman.git /openfhe-uniman && \
    cd /openfhe-uniman && \
    git checkout gpu-cuda && \
    mkdir build

# Set the working directory
WORKDIR /openfhe-uniman/build

# Configure and build the project
RUN cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc .. 
#	&& \    make

# Set the entry point
# CMD ["/bin/bash"]

