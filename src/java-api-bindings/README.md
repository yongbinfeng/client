<!--
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)



### JavaCPP and Java CPP Presets
The [JavaCPP Project](https://github.com/bytedeco/javacpp) is an open source project that provides efficient access to native C++ inside Java. It is used in the Tritonserver
project to generate Java bindings for Tritonserver.h. [JavaCPP-Presets repository](https://github.com/bytedeco/javacpp-presets) contains pre-built binaries for JavaCPP. To learn more about JavaCPP and JavaCPP-Presets, please visit [Bytedeco's homepage](http://bytedeco.org/).


Sample Usage
------------
Here is a simple example of Triton Inference Server ported to Java from the `simple.cc` sample file available at:

 * https://github.com/triton-inference-server/server/tree/main/src/servers

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `Simple.java` source files from the [`samples/`](samples/) subdirectory, simply execute on the command line:
```bash
 $ mvn compile exec:java -Dexec.args="-r /path/to/models"
```
This sample intends to show how to call the Java-mapped C API of Triton to execute inference requests.

### Steps to run this sample inside an NGC container

```bash
 $ docker run -it --gpus=all -v $(pwd):/workspace nvcr.io/nvidia/tritonserver:22.01-py3 bash
 $ apt update && apt install -y openjdk-11-jdk
 $ wget https://archive.apache.org/dist/maven/maven-3/3.8.4/binaries/apache-maven-3.8.4-bin.tar.gz
 $ tar zxvf apache-maven-3.8.4-bin.tar.gz
 $ export PATH=/opt/tritonserver/apache-maven-3.8.4/bin:$PATH
 $ git clone https://github.com/triton-inference-server/client.git
 $ cd client/src/java-api-bindings
 $ mvn clean install --projects .,tritonserver
 $ mvn clean install -f platform --projects ../tritonserver/platform -Djavacpp.platform=linux-x86_64
 $ cd java-api-bindings/build
 $ export JAVA_API_BINDINGS_PATH=$PWD
 $ mvn clean install -Djavacpp.platform=linux-x86_64
```

This sample is the Java implementation of the simple example written for the [C API](https://github.com/triton-inference-server/server/blob/main/docs/inference_protocols.md#c-api).

### Steps to run any binary linked to Triton Inference Server using JavaCPP inside an NGC container

To run your code, you will need to:

 1. Create `pom.xml` and `<your code>.java` source files, and
 2. Run your java file with
```bash
 $ java -cp ${JAVA_API_BINDINGS_PATH}/tritonserver-uber-<binding-version>.jar <your code>.java
```
