# Build Instructions with CUDA

## Requirements
We have implemented and tested using the following environment:
1. Cmake version 3.22
2. CUDA version 12.2

To successfully build with CUDA the following environment configurations are needed:

1. Locate the installation path of CUDAToolkit:
```commandline
which nvcc
```
should return something like this:
```commandline
/usr/local/cuda-12.2/bin/nvcc
```
The root folder is what we need ( i.e. `/usr/local/cuda-12.2`) .

2. Set environment variables:
```commandline
export CUDAToolkit_ROOT=/usr/local/cuda-12.2
export PATH=$CUDAToolkit_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAToolkit_ROOT/lib64:$LD_LIBRARY_PATH
```

3. Build the cmake project:
```commandline
cd openfhe-uniman/
cmake .
make
```

4. Run applications:
```commandline
./bin/examples/pke/simple-integers-bgvrns
```

## CLion
