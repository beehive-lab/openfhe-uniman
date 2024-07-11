#!/bin/bash
# This script builds a Docker image, runs a Docker container with GPU support, and executes a specific example inside the container automatically.

docker run --gpus all -it beehivelab/openfhe-uniman-gpu-cuda:latest ./build/bin/examples/pke/simple-integers-bgvrns

