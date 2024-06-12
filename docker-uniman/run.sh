#!/bin/bash
# This script builds a Docker image, runs a Docker container with GPU support, and executes a specific example inside the container automatically.

docker run --gpus all -it openfhe-uniman-gpu-cuda ./build/bin/examples/pke/simple-integers-bgvrns

