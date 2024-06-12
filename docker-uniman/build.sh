#!/bin/bash
# This script builds a Docker image, runs a Docker container with GPU support, and
# executes a specific example inside the container automatically.

# Build command:
docker build -t openfhe-uniman-gpu-cuda .
