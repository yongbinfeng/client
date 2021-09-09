// Copyright (c) 2021, NVIDIA CORPORATION & Affiliates. All rights reserved.
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
#pragma once

#include "load_manager.h"


namespace triton { namespace perfanalyzer {

class LoadManager {
 public:
 protected:
  /// Helper function to allocate and prepare local memory
  /// \return cb::Error object indicating success or failure.
  cb::Error InitLocalMemory();

  /// Helper function to prepare the InferContext for sending inference
  /// request for local memory.
  /// \param ctx The target InferContext object.
  /// \return cb::Error object indicating success or failure.
  cb::Error PrepareLocalMemoryInfer(InferContext* ctx);

  /// Helper function to unregister all allocated memory
  cb::Error UnregisterLocalMemory();

 private:
  /// Helper function to update the local memory inputs
  /// \param inputs The vector of pointers to InferInput objects
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return cb::Error object indicating success or failure.
  cb::Error SetInputsLocalMemory(
      const std::vector<cb::InferInput*>& inputs, const int stream_index,
      const int step_index);

 protected:
  bool using_local_memory_;
  // Map from shared memory key to its starting address and size
  std::unordered_map<std::string, std::pair<uint8_t*, size_t>>
      local_memory_regions_;
};

}}  // namespace triton::perfanalyzer
