# validate_triton_client
To validate the installation of Triton Client (C++ version)


## Introduction

Use the model from [Triton server tutorial with PyTorch](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/PyTorch).

### Usage
Start the server:
```bash
./scripts/start_triton_server.sh
```

Start the client
```bash
TRITON_IMAGE="docker.io/docexoty/tritonserver:latest" && podman-hpc run -it --gpu --rm --ipc=host --net=host \
  --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/ -w /workspace $TRITON_IMAGE bash

pip install torchvision==0.18.0

python models/resnet50/client.py
```

### Build the client
```bash
cmake -B build -S src -DCMAKE_INSTALL_PREFIX=/pscratch/sd/x/xju/athena_dev/triton_20250408_nosystemmd/install

cmake --build build -- install

./build/bin/test_resnet50  -u "login16:8001" -i models/resnet50/img1.txt 
```
The expected output from the above command is:
```text
[xju@login38] validate_triton_client >cmake --build  build -- install && ./build/bin/test_resnet50  -u "login16:8001" -i models/resnet50/img1.txt 
-- Found RE2 via CMake.
-- Configuring done (0.4s)
-- Generating done (0.1s)
-- Build files have been written to: /pscratch/sd/x/xju/code/validate_triton_client/build
[ 50%] Building CXX object CMakeFiles/test_resnet50.dir/test_resnet50.cxx.o
[100%] Linking CXX executable bin/test_resnet50
[100%] Built target test_resnet50
Install the project...
-- Install configuration: ""
-- Installing: /pscratch/sd/x/xju/athena_dev/triton_20250408_nosystemmd/install/bin/test_resnet50
-- Set non-toolchain portion of runtime path of "/pscratch/sd/x/xju/athena_dev/triton_20250408_nosystemmd/install/bin/test_resnet50" to ""
Reading input file: models/resnet50/img1.txt
total input entries: 150528
Input: [1.015926 1.033051 1.015926 1.015926 1.015926 ]
Expected Input: [1.015926  1.0330508 1.015926  1.015926  1.015926 ]
Output: [-0.257271 4.352922 -2.057417 -1.998921 -3.177508 ]
Expected: [-0.25864124  4.3533406  -2.0574622  -1.9991533  -3.178356]
```
