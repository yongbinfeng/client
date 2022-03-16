# Triton Client Libraries and Examples

Dummy triton client example to get it running with ExaTrk. Orignal code repo is [here](https://github.com/triton-inference-server/client).

## Compile the client libaries in the container
The client libraries need to be compiled with the ExaTrk docker container. Clone the client code first
```
mkdir testClient
git clone git@github.com:yongbinfeng/client.git -b ExaTrk
```

Start the docker container:
```
docker run -it --rm -v $PWD:$PWD -w $PWD --gpus all docexoty/exatrkx:tf2.5-torch1.9-cuda11.2-ubuntu20.04-rapids21.10-devel-hep bash
# remove protobuf
conda remove protobuf
apt-get install rapidjson-dev
apt-get install libb64-dev
```

The `rapidjson` and `libb64` packages are needed. The `protobuf` package in conda seems to have some conflicts with the Triton client modules so they are removed for now. Need to test its effects.

Compile the code inside the container:
```
cd testClient/clients
mkdir build
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_PERF_ANALYZER=ON -DTRITON_ENABLE_GPU=ON -DTRITON_ENABLE_EXAMPLES=ON ..
make -j 20 cc-clients
```

## Run the tests
Start the container with the model file
```
mkdir models
git clone git@github.com:yongbinfeng/TritonDemo.git .

```
