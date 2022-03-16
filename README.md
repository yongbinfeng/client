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
nvidia-docker run -it --gpus=1 -p8020:8000 -p8021:8001 -p8022:8002 --rm -v$PWD/models:/models nvcr.io/nvidia/tritonserver:21.02-py3 tritonserver --model-repository=/models
```
You should be able to see:
```
I0316 20:11:23.826043 145 server.cc:538]
+--------+---------+--------+
| Model  | Version | Status |
+--------+---------+--------+
| ExaTrk | 1       | READY  |
+--------+---------+--------+

I0316 20:11:23.829186 145 tritonserver.cc:1642]
+----------------------------------+------------------------------------------------------------------------------------------------------------------------+
| Option                           | Value                                                                                                                  |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------+
| server_id                        | triton                                                                                                                 |
| server_version                   | 2.7.0                                                                                                                  |
| server_extensions                | classification sequence model_repository schedule_policy model_configuration system_shared_memory cuda_shared_memory b |
|                                  | inary_tensor_data statistics                                                                                           |
| model_repository_path[0]         | /models                                                                                                                |
| model_control_mode               | MODE_NONE                                                                                                              |
| strict_model_config              | 1                                                                                                                      |
| pinned_memory_pool_byte_size     | 268435456                                                                                                              |
| cuda_memory_pool_byte_size{0}    | 67108864                                                                                                               |
| min_supported_compute_capability | 6.0                                                                                                                    |
| strict_readiness                 | 1                                                                                                                      |
| exit_timeout                     | 30                                                                                                                     |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------+

I0316 20:11:23.835548 145 grpc_server.cc:3979] Started GRPCInferenceService at 0.0.0.0:8001
I0316 20:11:23.835939 145 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I0316 20:11:23.878997 145 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002
```

On the client side, 
```
cd install/bin
./simple_grpc_async_infer_client
```

This should send some data to the server with GRPC requests and receive some outputs.
