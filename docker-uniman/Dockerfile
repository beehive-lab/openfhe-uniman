# Use the base NVIDIA CUDA image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install necessary dependencies including wget and curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    vim \
    git \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install CMake 3.25 or higher
RUN curl -L https://cmake.org/files/v3.25/cmake-3.25.0-linux-x86_64.tar.gz | \
    tar --strip-components=1 -xz -C /usr/local && \
    cmake --version

# Clone the repository and switch to the desired branch
RUN git clone https://github.com/beehive-lab/openfhe-uniman.git && \
    cd openfhe-uniman && \
    git checkout gpu-cuda && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make

# Set the working directory
WORKDIR /openfhe-uniman

# Define the entry point or command to execute when the container starts
# CMD ["/bin/bash"] # or specify your build or run command

# Build command:
# docker build -t openfhe-uniman-gpu-cuda .

# Run command:
# docker run --gpus all -it openfhe-uniman-gpu-cuda

# Inside the docker run the example:
# ./build/bin/examples/pke/simple-integers-bgvrns 

