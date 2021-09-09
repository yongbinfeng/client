// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "local_memory_manager.h"
#include "triton/core/tritonserver.h"


namespace triton { namespace perfanalyzer {
cb::Error
LoadManager::InitLocalMemory()
{
  using_local_memory_ = true;
  // Calling this function for the clean start
  backend_->UnregisterLocalMemory();
  // Allocate the shared memory for outputs
  for (const auto& output : *(parser_->Outputs())) {
    int64_t batch1_bytesize =
        ByteSize(output.second.shape_, output.second.datatype_);
    if (batch1_bytesize < 0) {
      batch1_bytesize = output_shm_size_;
    }
    void* output_allocated_ptr;
    size_t alloc_size = batch1_bytesize * batch_size_;
    std::string region_name(TensorToRegionName(output.first));
    output_allocated_ptr = malloc(alloc_size);
    local_memory_regions_[region_name] =
        std::pair<void*, size_t>(output_allocated_ptr, alloc_size);
    RETURN_IF_ERROR(backend_->RegisterLocalMemory(region_name, alloc_size));
  }

  // Allocate shared memory for inputs
  for (const auto& input : *(parser_->Inputs())) {
    for (int i = 0; i < (int)data_loader_->GetDataStreamsCount(); i++) {
      for (int j = 0; j < (int)data_loader_->GetTotalSteps(i);
           j += batch_size_) {
        // Extract the data for requested batch size
        std::vector<const uint8_t*> data_ptrs;
        std::vector<size_t> byte_size;
        size_t alloc_size = 0;
        size_t count = 0;
        size_t max_count = input.second.is_shape_tensor_ ? 1 : batch_size_;
        std::vector<int64_t> shape;
        std::vector<int64_t> prev_shape;
        while (count < max_count) {
          const uint8_t* data_ptr;
          size_t batch1_bytesize;

          RETURN_IF_ERROR(data_loader_->GetInputShape(
              input.second, i, (j + count) % data_loader_->GetTotalSteps(i),
              &shape));
          if (!shape.empty()) {
            if (count == 0) {
              prev_shape = shape;
            } else {
              if (!std::equal(shape.begin(), shape.end(), prev_shape.begin())) {
                return cb::Error(
                    "can not batch tensors with different shapes together "
                    "(input '" +
                    input.first + "' expected shape " +
                    ShapeVecToString(prev_shape) + " and received " +
                    ShapeVecToString(shape));
              }
            }
          }
          RETURN_IF_ERROR(data_loader_->GetInputData(
              input.second, i, (j + count) % data_loader_->GetTotalSteps(i),
              &data_ptr, &batch1_bytesize));
          data_ptrs.push_back(data_ptr);
          byte_size.push_back(batch1_bytesize);
          alloc_size += batch1_bytesize;
          count++;
        }
        // Validate if the shape tensors specified in the batch are identical.
        while (count < batch_size_) {
          const uint8_t* data_ptr;
          size_t batch1_bytesize;
          RETURN_IF_ERROR(data_loader_->GetInputData(
              input.second, i, (j + count) % data_loader_->GetTotalSteps(i),
              &data_ptr, &batch1_bytesize));
          if (batch1_bytesize != byte_size.back()) {
            return cb::Error(
                "The shape tensors should be identical in a batch (mismatch in "
                "size)");
          }

          for (size_t data_idx = 0; data_idx < batch1_bytesize; data_idx++) {
            if (*(data_ptr + data_idx) != *(data_ptrs.back() + data_idx)) {
              return cb::Error(
                  "The shape tensors should be identical in a batch (mismatch "
                  "in content)");
            }
          }
          count++;
        }
        // Generate the shared memory region name
        std::string region_name(
            TensorToRegionName(input.first) + "_" + std::to_string(i) + "_" +
            std::to_string(j));

        void* input_allocated_ptr;
        input_allocated_ptr = malloc(alloc_size);
        local_memory_regions_[region_name] =
            std::pair<void*, size_t>(input_allocated_ptr, alloc_size);
      }
    }
  }

  return cb::Error::Success;
}

cb::Error
LoadManager::PrepareLocalMemoryInfer(InferContext* ctx)
{
  return cb::Error::Success;
}


cb::Error
LoadManager::SetInputsLocalMemory(
    const std::vector<cb::InferInput*>& inputs, const int stream_index,
    const int step_index)
{
  return cb::Error::Success;
}

cb::Error
LoadManager::UnregisterLocalMemory()
{
}
}}  // namespace triton::perfanalyzer
