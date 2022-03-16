// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <unistd.h>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include "grpc_client.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace {

void
ValidateShapeAndDatatype(
    const std::string& name, std::shared_ptr<tc::InferResult> result)
{
  std::vector<int64_t> shape;
  FAIL_IF_ERR(
      result->Shape(name, &shape), "unable to get shape for '" + name + "'");
  // Validate shape
  if ((shape.size() != 2) || (shape[0] != 1) || (shape[1] != 8)) {
    std::cerr << "error: received incorrect shapes for '" << name << "'"
              << std::endl;
    exit(1);
  }
  std::string datatype;
  FAIL_IF_ERR(
      result->Datatype(name, &datatype),
      "unable to get datatype for '" + name + "'");
  // Validate datatype
  if (datatype.compare("FP32") != 0) {
    std::cerr << "error: received incorrect datatype for '" << name
              << "': " << datatype << std::endl;
    exit(1);
  }
}

void
ValidateResult(
    const std::shared_ptr<tc::InferResult> result,
    std::vector<float>& input0_data)
{
  // Validate the results...
  ValidateShapeAndDatatype("embedding_output", result);

  // Get pointers to the result returned...
  float* output0_data;
  size_t output0_byte_size;
  FAIL_IF_ERR(
      result->RawData(
          "embedding_output", (const uint8_t**)&output0_data, &output0_byte_size),
      "unable to get result data for 'embedding_output'");
  if (output0_byte_size != 32) {
    std::cerr << "error: received incorrect byte size for 'embedding_output': "
              << output0_byte_size << std::endl;
    exit(1);
  }

  std::cout << "inputs " << std::endl;
  for (size_t i = 0; i < 3; ++i) {
    std::cout << i << " input " << input0_data[i] << std::endl;
  }

  std::cout << "outputs " << std::endl;
  for (size_t i = 0; i < 8; ++i) {
    std::cout << i << " output " << *(output0_data + i) << std::endl;
  }

  // Get full response
  std::cout << result->DebugString() << std::endl;
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8021");
  tc::Headers http_headers;
  uint32_t client_timeout = 0;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:t:H:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 't':
        client_timeout = std::stoi(optarg);
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        http_headers[header] = arg.substr(header.size() + 1);
        break;
      }
      case '?':
        Usage(argv);
        break;
    }
  }

  std::string model_name = "ExaTrk";
  std::string model_version = "";

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  FAIL_IF_ERR(
      tc::InferenceServerGrpcClient::Create(&client, url, verbose),
      "unable to create grpc client");

  // Create the data for the input tensors.
  std::vector<float> input0_data(3);
  for (size_t i = 0; i < 3; ++i) {
    input0_data[i] = i;
  }

  std::vector<int64_t> shape{1, 3};

  // Initialize the inputs with the data.
  tc::InferInput* input0;

  FAIL_IF_ERR(
      tc::InferInput::Create(&input0, "sp_features", shape, "FP32"),
      "unable to get INPUT0");
  std::shared_ptr<tc::InferInput> input0_ptr;
  input0_ptr.reset(input0);

  FAIL_IF_ERR(
      input0_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input0_data[0]),
          input0_data.size() * sizeof(float)),
      "unable to set data for INPUT0");

  // Generate the outputs to be requested.
  tc::InferRequestedOutput* output0;

  FAIL_IF_ERR(
      tc::InferRequestedOutput::Create(&output0, "embedding_output"),
      "unable to get 'embedding_output'");
  std::shared_ptr<tc::InferRequestedOutput> output0_ptr;
  output0_ptr.reset(output0);


  // The inference settings. Will be using default for now.
  tc::InferOptions options(model_name);
  options.model_version_ = model_version;
  options.client_timeout_ = client_timeout;

  std::vector<tc::InferInput*> inputs = {input0_ptr.get()};
  std::vector<const tc::InferRequestedOutput*> outputs = {output0_ptr.get()};

  // Send inference request to the inference server.
  std::mutex mtx;
  std::condition_variable cv;
  size_t repeat_cnt = 2;
  size_t done_cnt = 0;
  for (size_t i = 0; i < repeat_cnt; i++) {
    FAIL_IF_ERR(
        client->AsyncInfer(
            [&, i](tc::InferResult* result) {
              {
                std::shared_ptr<tc::InferResult> result_ptr;
                result_ptr.reset(result);
                std::lock_guard<std::mutex> lk(mtx);
                std::cout << "Callback no." << i << " is called" << std::endl;
                done_cnt++;
                if (result_ptr->RequestStatus().IsOk()) {
                  ValidateResult(result_ptr, input0_data);
                } else {
                  std::cerr << "error: Inference failed: "
                            << result_ptr->RequestStatus() << std::endl;
                  exit(1);
                }
              }
              cv.notify_all();
            },
            options, inputs, outputs, http_headers),
        "unable to run model");
  }

  // Wait until all callbacks are invoked
  {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&]() {
      if (done_cnt >= repeat_cnt) {
        return true;
      } else {
        return false;
      }
    });
  }
  if (done_cnt == repeat_cnt) {
    std::cout << "All done" << std::endl;
  } else {
    std::cerr << "Done cnt: " << done_cnt
              << " does not match repeat cnt: " << repeat_cnt << std::endl;
    exit(1);
  }

  // Send another AsyncInfer whose callback defers the completed request
  // to another thread (main thread) to handle
  bool callback_invoked = false;
  std::shared_ptr<tc::InferResult> result_placeholder;
  FAIL_IF_ERR(
      client->AsyncInfer(
          [&](tc::InferResult* result) {
            {
              std::shared_ptr<tc::InferResult> result_ptr;
              result_ptr.reset(result);
              // Defer the response retrieval to main thread
              std::lock_guard<std::mutex> lk(mtx);
              callback_invoked = true;
              result_placeholder = std::move(result_ptr);
            }
            cv.notify_all();
          },
          options, inputs, outputs, http_headers),
      "unable to run model");

  // Ensure callback is completed
  {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&]() { return callback_invoked; });
  }

  // Get deferred response
  std::cout << "Getting results from deferred response" << std::endl;
  if (result_placeholder->RequestStatus().IsOk()) {
    ValidateResult(result_placeholder, input0_data);
  } else {
    std::cerr << "error: Inference failed: "
              << result_placeholder->RequestStatus() << std::endl;
    exit(1);
  }

  tc::InferStat infer_stat;
  client->ClientInferStat(&infer_stat);
  std::cout << "completed_request_count " << infer_stat.completed_request_count
            << std::endl;
  std::cout << "cumulative_total_request_time_ns "
            << infer_stat.cumulative_total_request_time_ns << std::endl;
  std::cout << "cumulative_send_time_ns " << infer_stat.cumulative_send_time_ns
            << std::endl;
  std::cout << "cumulative_receive_time_ns "
            << infer_stat.cumulative_receive_time_ns << std::endl;

  std::cout << "PASS : Async Infer" << std::endl;

  return 0;
}
