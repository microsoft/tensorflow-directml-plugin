/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/c/logging.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tfdml/core/util/macros.h"
#include "tfdml/core/util/op_kernel_context.h"
#include "tfdml/core/util/refcount.h"
#include "tfdml/core/util/status.h"

namespace tfdml {

// A ResourceMgr instance keeps track of named and typed resources
// grouped into containers.
//
// Each resource must be represented as a sub-class of ResourceBase,
// which is reference counted explicitly.  Each named resource is
// registered with ResourceMgr under a named "container" name. At any
// time, there is at most one instance of a resource given the container
// name, the resource type and the resource name.
//
// All resources for a given container can be dropped by one call of
// Cleanup().
//
// E.g.,
//   struct MyVar : public ResourceBase {
//     mutex mu;
//     Tensor val;
//   }
//
//   ResourceMgr rm;
//
//   // Create a var.
//   MyVar* my_var = new MyVar;
//   my_var->val = Tensor(DT_FLOAT, my_shape);
//   my_var->val.flat<float>().setZeros();   // 0 initialized.
//   ctx->SetStatus(rm.Create("my_container", "my_name", my_var));
//
//   // += a variable.
//   MyVar* my_var = nullptr;
//   Status s = rm.Lookup("my_container", "my_name", &my_var);
//   if (s.ok()) {
//     my_var->val.flat<float>() += grad;
//   }
//   my_var->Unref();   // Or use ScopedUnref().
//   ctx->SetStatus(s);
class ResourceBase : public RefCounted {
 public:
  // Returns a debug string for *this.
  virtual std::string DebugString() const = 0;

  // Returns memory used by this resource.
  virtual int64_t MemoryUsed() const { return 0; }
};

// Container used for per-step resources.
class ScopedStepContainer {
 public:
  // step_id: the unique ID of this step. Doesn't have to be sequential, just
  // has to be unique.
  // cleanup: callback to delete a container of this name.
  // prefix: optional string prefix to disambiguate step containers.
  ScopedStepContainer(const int64_t step_id,
                      std::function<void(const std::string&)> cleanup)
      : name_(absl::StrCat("__per_step_", step_id)),
        step_id_(step_id),
        cleanup_(cleanup) {}

  ScopedStepContainer(const int64_t step_id,
                      std::function<void(const std::string&)> cleanup,
                      const std::string& prefix)
      : name_(absl::StrCat("__", prefix, "_per_step_", step_id)),
        step_id_(step_id),
        cleanup_(cleanup) {}

  ~ScopedStepContainer() { cleanup_(name_); }

  const std::string& name() const { return name_; }
  const int64_t step_id() const { return step_id_; }

 private:
  const std::string name_;
  const int64_t step_id_;
  const std::function<void(const std::string&)> cleanup_;
};

template <typename T>
void CheckDeriveFromResourceBase() {
  static_assert(std::is_base_of<ResourceBase, T>::value,
                "T must derive from ResourceBase");
}

class ResourceMgr {
 public:
  ResourceMgr();
  explicit ResourceMgr(const std::string& default_container);
  ~ResourceMgr();

  // Returns the default container name for *this.
  const std::string& default_container() const { return default_container_; }

  // Deletes the resource "name" from the "container".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  template <typename T>
  Status Delete(const std::string& container, const std::string& name);

  // Deletes the resource pointed by "handle".
  Status Delete(const tensorflow::ResourceHandleProto& handle);

  // Deletes all resources from the "container" and removes the container.
  Status Cleanup(const std::string& container);

  // Deletes all resources in all containers.
  void Clear();

  // If "container" has a resource "name", returns it in "*resource" and
  // the caller takes the ownership of one ref on "*resource".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr
  template <typename T, bool use_dynamic_cast = false>
  Status Lookup(const std::string& container, const std::string& name,
                T** resource) const;

  template <typename T, bool use_dynamic_cast = false>
  Status LookupOrCreate(const std::string& container, const std::string& name,
                        T** resource, std::function<Status(T**)> creator) {
    CheckDeriveFromResourceBase<T>();
    *resource = nullptr;
    Status s;
    {
      std::shared_lock<std::shared_mutex> l(mu_);
      s = LookupInternal<T, use_dynamic_cast>(container, name, resource);
      if (s.ok()) return s;
    }
    std::unique_lock<std::shared_mutex> l(mu_);
    s = LookupInternal<T, use_dynamic_cast>(container, name, resource);
    if (s.ok()) return s;
    TF_RETURN_IF_ERROR(creator(resource));
    s = DoCreate(container, std::type_index(typeid(T)), name, *resource);
    if (!s.ok()) {
      return errors::Internal("LookupOrCreate failed unexpectedly");
    }
    (*resource)->Ref();
    return s;
  }

 private:
  typedef std::pair<uint64_t, std::string> Key;
  typedef absl::flat_hash_map<Key, ResourceBase*> Container;

  const std::string default_container_;
  mutable std::shared_mutex mu_;
  std::unordered_map<std::string, Container*> containers_;

  Status DoDelete(const std::string& container, uint64_t type_hash_code,
                  const std::string& resource_name,
                  const std::string& type_name);
  Status DoDelete(const std::string& container, std::type_index type,
                  const std::string& resource_name);
  Status DoLookup(const std::string& container, std::type_index type,
                  const std::string& name, ResourceBase** resource) const;

  template <typename T, bool use_dynamic_cast = false>
  Status LookupInternal(const std::string& container, const std::string& name,
                        T** resource) const;

  Status DoCreate(const std::string& container, std::type_index type,
                  const std::string& name, ResourceBase* resource);

  // Inserts the type name for 'hash_code' into the hash_code to type name
  // map.
  Status InsertDebugTypeName(uint64_t hash_code, const std::string& type_name);

  // Returns the type name for the 'hash_code'.
  // Returns "<unknown>" if a resource with such a type was never inserted into
  // the container.
  const char* DebugTypeName(uint64_t hash_code) const;

  // Map from type hash_code to type name.
  std::unordered_map<uint64_t, std::string> debug_type_names_;

  ResourceMgr(const ResourceMgr&) = delete;
  void operator=(const ResourceMgr&) = delete;
};

template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx,
                              const tensorflow::ResourceHandleProto& p,
                              T** value, std::function<Status(T**)> creator) {
  return ctx->resource_manager()->LookupOrCreate(p.container(), p.name(), value,
                                                 creator);
}

template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx,
                              const tensorflow::ResourceHandleProto& p,
                              RefCountPtr<T>* value,
                              std::function<Status(T**)> creator) {
  T* raw_ptr = nullptr;
  TF_RETURN_IF_ERROR(LookupOrCreateResource<T>(ctx, p, &raw_ptr, creator));
  value->reset(raw_ptr);

  return Status::OK();
}

// Looks up a resource pointed by a given resource handle.
//
// If the lookup is successful, the caller takes the ownership of one ref on
// `*value`, and must call its `Unref()` method when it has finished using it.
template <typename T, bool use_dynamic_cast = false>
Status LookupResource(OpKernelContext* ctx,
                      const tensorflow::ResourceHandleProto& p, T** value);

// Looks up a resource pointed by a given resource handle.
//
// Prefer usage of LookupResource taking `core::RefCountPtr` to avoid
// requiring the caller to explicitly call `Unref()`.
template <typename T>
Status LookupResource(OpKernelContext* ctx,
                      const tensorflow::ResourceHandleProto& p,
                      RefCountPtr<T>* value);

// This class is used to guarantee that an anonymous resource is deleted
// (irrespective of whether a resource deleter op is called explicitly or
// the execution encounters an error before the op runs).
//
// This is achieved by wrapping an instance of this class into a variant
// tensor which is passed as an input to a resource deleter op. If the
// execution encounters an error before the op runs, the tensor will be
// destroyed, essentially triggering the iterator deletion.
// NOTE: This is not a feature-complete implementation of the DT_VARIANT
// specification. In particular, we cannot serialize the `ResourceMgr`
// object, so the `Encode()` and `Decode()` methods are not implemented.
class ResourceDeleter {
 public:
  ResourceDeleter() : deleter_() {}

  ResourceDeleter(tensorflow::ResourceHandleProto handle,
                  ResourceMgr* resource_manager)
      : deleter_(std::make_shared<Helper>(handle, resource_manager)) {}

  ResourceDeleter(ResourceDeleter&& rhs) : deleter_(std::move(rhs.deleter_)) {
    TF_VLog(3, "ResourceDeleter move constructor called.");
  }

  ResourceDeleter(const ResourceDeleter& rhs) : deleter_(rhs.deleter_) {
    TF_VLog(3, "ResourceDeleter copy constructor called.");
  }

  ResourceDeleter& operator=(const ResourceDeleter& rhs) = delete;

  ResourceDeleter& operator=(ResourceDeleter&& rhs) = default;

  virtual ~ResourceDeleter() {
    TF_VLog(3, "ResourceDeleter destructor called.");
  }

 private:
  // Helper that performs reference counting for the parent class and deletes
  // the iterator resource when the refcount goes to zero.
  //
  // NOTE: The object is borrowing a pointer to the resource manager.
  // Consequently, the tensor containing this object should not escape the
  // function in which was created (so that it is guaranteed that the resource
  // manager will outlive it).
  struct Helper {
    Helper(tensorflow::ResourceHandleProto handle,
           ResourceMgr* resource_manager)
        : handle(handle), resource_manager(resource_manager) {}

    Helper(const Helper& rhs) = delete;
    Helper(Helper&& rhs) = delete;

    ~Helper() {
      TF_VLog(3, "Deleting Resource: %s", handle.DebugString().c_str());
      resource_manager->Delete(handle);
    }

    tensorflow::ResourceHandleProto handle;
    ResourceMgr* resource_manager;  // not owned
  };

  std::shared_ptr<Helper> deleter_;
};

// Implementation details below.

// Simple wrapper to allow conditional dynamic / static casts.
template <typename T, bool use_dynamic_cast>
struct TypeCastFunctor {
  static T* Cast(ResourceBase* r) { return static_cast<T*>(r); }
};

template <typename T>
struct TypeCastFunctor<T, true> {
  static T* Cast(ResourceBase* r) { return dynamic_cast<T*>(r); }
};

template <typename T>
Status ResourceMgr::Delete(const std::string& container,
                           const std::string& name) {
  CheckDeriveFromResourceBase<T>();
  return DoDelete(container, std::type_index(typeid(T)), name);
}

template <typename T, bool use_dynamic_cast>
Status ResourceMgr::LookupInternal(const std::string& container,
                                   const std::string& name,
                                   T** resource) const {
  ResourceBase* found = nullptr;
  Status s = DoLookup(container, std::type_index(typeid(T)), name, &found);
  if (s.ok()) {
    // It's safe to down cast 'found' to T* since
    // typeid(T).hash_code() is part of the map key.
    *resource = TypeCastFunctor<T, use_dynamic_cast>::Cast(found);
  }
  return s;
}

template <typename T, bool use_dynamic_cast>
Status ResourceMgr::Lookup(const std::string& container,
                           const std::string& name, T** resource) const {
  CheckDeriveFromResourceBase<T>();
  std::shared_lock<std::shared_mutex> l(mu_);
  return LookupInternal<T, use_dynamic_cast>(container, name, resource);
}

template <typename T, bool use_dynamic_cast>
Status LookupResource(OpKernelContext* ctx,
                      const tensorflow::ResourceHandleProto& p, T** value) {
  return ctx->resource_manager()->Lookup<T, use_dynamic_cast>(p.container(),
                                                              p.name(), value);
}

template <typename T>
Status LookupResource(OpKernelContext* ctx,
                      const tensorflow::ResourceHandleProto& p,
                      RefCountPtr<T>* value) {
  T* raw_ptr = nullptr;
  TF_RETURN_IF_ERROR(LookupResource<T, false>(ctx, p, &raw_ptr));
  value->reset(raw_ptr);

  return Status::OK();
}

const tensorflow::ResourceHandleProto& HandleFromInput(OpKernelContext* ctx,
                                                       int input);

}  //  namespace tfdml
