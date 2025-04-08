# validate_triton_client
To validate the installation of Triton Client (C++ version)


## Introduction

Use the model from [Triton server tutorial with PyTorch](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/PyTorch).

### Usage

Start the client
```bash
TRITON_IMAGE="docker.io/docexoty/tritonserver:latest" && podman-hpc run -it --gpu --rm --ipc=host --net=host \
  --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/ -w /workspace $TRITON_IMAGE bash

pip install torchvision==0.18.0
```

### Build the client
```bash
cmake -B build -S src -DCMAKE_INSTALL_PREFIX=/pscratch/sd/x/xju/athena_dev/triton_20250408_nosystemmd/install

cmake --build build -- install

./build/bin/test_resnet50  -u "login16:8001" -i models/resnet50/img1.txt 
```
