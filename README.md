# validate_triton_client
To validate the installation of Triton Client (C++ version)


## Introduction

Use the model from [Triton server tutorial with PyTorch](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/PyTorch).

### Usage

#### Start the server:
```bash
./scripts/start-tritonserver.sh
```

#### Build the client

```bash
cd /pscratch/sd/x/xju/athena_dev/triton_20250408_nosystemmd
cmake -B build -S client-r24.12/src/c++ \
     -DBUILD_SHARED_LIBS=ON \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CXX_STANDARD=20 \
     -DCMAKE_INSTALL_PREFIX=${PWD}/install \
     -DTRITON_ENABLE_CC_GRPC=ON \
     -DTRITON_ENABLE_CC_HTTP=OFF \
     -DTRITON_ENABLE_TESTS=OFF \
     -DTRITON_ENABLE_ZLIB=OFF \
     -DTRITON_USE_THIRD_PARTY=OFF \
     -DTRITON_REPO_ORGANIZATION=https://github.com/triton-inference-server \
     -DTRITON_KEEP_TYPEINFO=ON
```

#### Build the client and run the test
```bash
cmake -B build -S src -DCMAKE_INSTALL_PREFIX=/pscratch/sd/x/xju/athena_dev/triton_20250408_nosystemmd/install

cmake --build build -- install

./build/bin/test_resnet50  -u "login16:8001" -i models/resnet50/img1.txt 
```
The expected output from the above command is:
```text
[xju@login38] validate_triton_client >cmake --build  build -- install && ./build/bin/test_resnet50  -u "login16:8001" -i models/resnet50/img1.txt 
[ 50%] Building CXX object CMakeFiles/test_resnet50.dir/test_resnet50.cxx.o
[100%] Linking CXX executable bin/test_resnet50
[100%] Built target test_resnet50
Install the project...
-- Install configuration: ""
-- Installing: /pscratch/sd/x/xju/athena_dev/triton_20250408_nosystemmd/install/bin/test_resnet50
-- Set non-toolchain portion of runtime path of "/pscratch/sd/x/xju/athena_dev/triton_20250408_nosystemmd/install/bin/test_resnet50" to ""
Reading input file: models/resnet50/img1.txt
total input entries: 150528
Input values: [1.015926 1.033051 1.015926 1.015926 1.015926 ]
Expected input: [1.015926 1.033051 1.015926 1.015926 1.015926]
Input values match expected values within 0.01
Output values: [-0.257271 4.352922 -2.057417 -1.998921 -3.177508 ]
Expected output: [-0.258641 4.353341 -2.057462 -1.999153 -3.178356]
Output values match expected values within 0.01
```

Start the client
```bash
TRITON_IMAGE="docker.io/docexoty/tritonserver:latest" && podman-hpc run -it --gpu --rm --ipc=host --net=host \
  --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/ -w /workspace $TRITON_IMAGE bash

pip install torchvision==0.18.0

python models/resnet50/client.py
```
