/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Portions Copyright (c) Microsoft Corporation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "tfdml/core/common_runtime/dml/dml_common.h"
#include "tfdml/core/common_runtime/dml/dml_device.h"
#include "tfdml/core/common_runtime/dml/dml_kernel_manager.h"
#include "tfdml/core/kernels/dml_ops_common.h"
#include "tfdml/core/util/op_kernel.h"

namespace tfdml {

class ShapeHelper;
class InitializationHelper;
class OpKernelConstruction;

enum class DmlKernelCachePolicy {
  Always,
  Never,
  Default = Always,
};

// Wraps a DmlKernel and implements the OpKernel interface on its behalf. This
// wrapper forms the boundary between the DML kernel implementations and
// tensorflow's kernel interfaces, and presents a simpler abstraction to the
// wrapped DmlKernel than what is supplied in the full OpKernel.
class DmlKernelWrapperBase : public OpKernel {
 public:
  explicit DmlKernelWrapperBase(DmlKernelCachePolicy cache_policy,
                                const char* op_type_string,
                                const char* op_name);
  virtual ~DmlKernelWrapperBase() = default;

  void Compute(OpKernelContext* raw_ctx);

  virtual std::shared_ptr<const BaseAttributes> GetAttributes() const = 0;

 protected:
  virtual const ShapeHelper* GetShapeHelper() const = 0;
  virtual std::shared_ptr<const InitializationHelper>
  CreateInitializationHelper(OpKernelContext* ctx) const = 0;

  virtual std::shared_ptr<DmlKernel> CreateCachedKernel(
      DmlKernelConstruction* ctx, const DmlKernelManager& kernel_manager,
      const DmlKernelKey& key,
      const InitializationHelper* initialized_helper) const = 0;

  virtual std::shared_ptr<DmlKernel> TryGetCachedKernel(
      const DmlKernelManager& kernel_manager,
      const DmlKernelKey& key) const = 0;

  virtual std::shared_ptr<DmlKernel> CreateKernel(
      DmlKernelConstruction* ctx,
      const InitializationHelper* initialized_helper) const = 0;

  virtual StatusOr<DmlGpuEvent> ComputeKernel(
      DmlKernel* kernel, DmlKernelContext* context) const = 0;

 protected:
  // Creates a key which uniquely identifies the kernel instance we need. The
  // returned key can be used to retrieve an appropriate DML kernel from the
  // cache.
  virtual DmlKernelKey CreateKernelKey(OpKernelContext* ctx) const;

  DmlKernelCachePolicy cache_policy_;
};

// Implements a (templated) GetOrCreateKernel and output shape computation for
// the kernel wrapper.
//
// `TKernel` must be a DmlKernel implementation.
// `TShapeHelper` must be a type that matches the following signature:
//     struct S {
//       S(OpKernelConstruction* ctx); // constructor
//
//       // Computes the shapes of each output tensor. Must be thread-safe.
//       std::vector<TensorShape> GetOutputShapes(OpKernelContext* ctx) const;
//     }
//
// This class is intended to be used when registering DML kernels for an op.
// Example:
//     REGISTER_KERNEL_BUILDER(Name("Add").Device(DEVICE_DML),
//                             DmlKernelWrapper<DmlAddKernel, AddShapeHelper>);
//
template <typename TKernel, typename TShapeHelper,
          DmlKernelCachePolicy cache_policy = DmlKernelCachePolicy::Default>
class DmlKernelWrapper : public DmlKernelWrapperBase {
 public:
  using Attributes = typename TKernel::InitHelper::Attributes;

  explicit DmlKernelWrapper(OpKernelConstruction* ctx,
                            const char* op_type_string, const char* op_name)
      : DmlKernelWrapperBase(cache_policy, op_type_string, op_name),
        attr_(std::make_shared<Attributes>(ctx)) {
    // This is a (hopefully) temporary workaround since the API doesn't expose
    // any way to know whether a tensor is allocated with host or device memory,
    // and no way to map the tensor name to its index.
    // TODO: Remove this when/if the following PR gets merged
    // https://github.com/tensorflow/tensorflow/pull/51759
    for (int host_input_index : TKernel::host_input_indices) {
      if (host_input_index >= input_memory_types_.size()) {
        input_memory_types_.resize(host_input_index + 1, DEVICE_MEMORY);
      }

      input_memory_types_[host_input_index] = HOST_MEMORY;
    }

    for (int host_output_index : TKernel::host_output_indices) {
      if (host_output_index >= output_memory_types_.size()) {
        output_memory_types_.resize(host_output_index + 1, DEVICE_MEMORY);
      }

      output_memory_types_[host_output_index] = HOST_MEMORY;
    }
  }

 protected:
  const ShapeHelper* GetShapeHelper() const final { return &shape_helper_; }

  std::shared_ptr<DmlKernel> CreateCachedKernel(
      DmlKernelConstruction* ctx, const DmlKernelManager& kernel_manager,
      const DmlKernelKey& key,
      const InitializationHelper* initialized_helper) const final {
    // If the cache policy is "Never", the kernel wrapper should simply create
    // the kernel directly instead of delegating to the kernel manager
    assert(cache_policy != DmlKernelCachePolicy::Never);

    // Create the kernel and cache it
    return kernel_manager.CreateCachedKernel<TKernel>(
        ctx, key,
        static_cast<const typename TKernel::InitHelper*>(initialized_helper));
  }

  std::shared_ptr<DmlKernel> TryGetCachedKernel(
      const DmlKernelManager& kernel_manager,
      const DmlKernelKey& key) const final {
    // If the cache policy is "Never", the kernel wrapper should never try to
    // retrieved a cached kernel
    assert(cache_policy != DmlKernelCachePolicy::Never);

    // Retrieve the kernel from the cache
    return kernel_manager.TryGetCachedKernel<TKernel>(key);
  }

  std::shared_ptr<DmlKernel> CreateKernel(
      DmlKernelConstruction* ctx,
      const InitializationHelper* initialized_helper) const final {
    return std::make_shared<TKernel>(
        ctx,
        static_cast<const typename TKernel::InitHelper*>(initialized_helper));
  }

  std::shared_ptr<const InitializationHelper> CreateInitializationHelper(
      OpKernelContext* ctx) const final {
    return std::make_shared<const typename TKernel::InitHelper>(ctx, attr_);
  }

  StatusOr<DmlGpuEvent> ComputeKernel(DmlKernel* kernel,
                                      DmlKernelContext* context) const final {
    return kernel->Compute(context);
  }

  std::shared_ptr<const BaseAttributes> GetAttributes() const { return attr_; }

  MemoryType input_memory_type(int index) const final {
    if (index >= input_memory_types_.size()) {
      return DEVICE_MEMORY;
    }
    return input_memory_types_[index];
  }

  MemoryType output_memory_type(int index) const final {
    if (index >= output_memory_types_.size()) {
      return DEVICE_MEMORY;
    }
    return output_memory_types_[index];
  }

 private:
  const std::shared_ptr<const Attributes> attr_;
  const TShapeHelper shape_helper_;
  absl::InlinedVector<MemoryType, 4> input_memory_types_;
  absl::InlinedVector<MemoryType, 1> output_memory_types_;
};

}  // namespace tfdml