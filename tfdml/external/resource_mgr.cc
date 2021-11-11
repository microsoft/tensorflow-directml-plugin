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

#include "resource_mgr.h"

#include <atomic>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tfdml/external/macros.h"
#include "tfdml/external/op_kernel_context.h"

namespace tfdml
{

Status ResourceMgr::InsertDebugTypeName(
    uint64_t hash_code,
    const std::string& type_name)
{
    auto iter = debug_type_names_.emplace(hash_code, type_name);
    if (iter.first->second != type_name)
    {
        return errors::AlreadyExists(
            "Duplicate hash code found for type ",
            type_name);
    }
    return Status::OK();
}

const char* ResourceMgr::DebugTypeName(uint64_t hash_code) const
{
    auto type_name_iter = debug_type_names_.find(hash_code);
    if (type_name_iter == debug_type_names_.end())
    {
        return "<unknown>";
    }
    else
    {
        return type_name_iter->second.c_str();
    }
}

ResourceMgr::ResourceMgr() : default_container_("localhost") {}

ResourceMgr::ResourceMgr(const std::string& default_container)
    : default_container_(default_container)
{
}

ResourceMgr::~ResourceMgr() { Clear(); }

void ResourceMgr::Clear()
{
    // We do the deallocation outside of the lock to avoid a potential deadlock
    // in case any of the destructors access the resource manager.
    std::unordered_map<std::string, Container*> tmp_containers;
    {
        std::unique_lock<std::shared_mutex> l(mu_);
        tmp_containers = std::move(containers_);
    }
    for (const auto& p : tmp_containers)
    {
        for (const auto& q : *p.second)
        {
            q.second->Unref();
        }
        delete p.second;
    }
    tmp_containers.clear();
}

Status ResourceMgr::DoDelete(
    const std::string& container,
    uint64_t type_hash_code,
    const std::string& resource_name,
    const std::string& type_name)
{
    ResourceBase* base = nullptr;
    {
        std::unique_lock<std::shared_mutex> l(mu_);
        auto it = containers_.find(container);
        Container* b = it == containers_.end() ? nullptr : it->second;
        if (b == nullptr)
        {
            return errors::NotFound(
                "Container ",
                container,
                " does not exist.");
        }
        auto iter = b->find({type_hash_code, resource_name});
        if (iter == b->end())
        {
            return errors::NotFound(
                "Resource ",
                container,
                "/",
                resource_name,
                "/",
                type_name,
                " does not exist.");
        }
        base = iter->second;
        b->erase(iter);
    }
    CHECK(base != nullptr);
    base->Unref();
    return Status::OK();
}

Status ResourceMgr::DoDelete(
    const std::string& container,
    std::type_index type,
    const std::string& resource_name)
{
    return DoDelete(container, type.hash_code(), resource_name, type.name());
}

Status ResourceMgr::Delete(const tensorflow::ResourceHandleProto& handle)
{
    return DoDelete(
        handle.container(),
        handle.hash_code(),
        handle.name(),
        "<unknown>");
}

Status ResourceMgr::DoLookup(
    const std::string& container,
    std::type_index type,
    const std::string& name,
    ResourceBase** resource) const
{
    auto it = containers_.find(container);
    const Container* b = it == containers_.end() ? nullptr : it->second;
    if (b == nullptr)
    {
        return errors::NotFound(
            "Container ",
            container,
            " does not exist. (Could not find resource: ",
            container,
            "/",
            name,
            ")");
    }
    auto iter = b->find({type.hash_code(), name});
    auto r = iter == b->end() ? nullptr : iter->second;
    if (r == nullptr)
    {
        return errors::NotFound(
            "Resource ",
            container,
            "/",
            name,
            "/",
            type.name(),
            " does not exist.");
    }
    *resource = const_cast<ResourceBase*>(r);
    (*resource)->Ref();
    return Status::OK();
}

Status ResourceMgr::DoCreate(
    const std::string& container,
    std::type_index type,
    const std::string& name,
    ResourceBase* resource)
{
    Container** b = &containers_[container];
    if (*b == nullptr)
    {
        *b = new Container;
    }
    if ((*b)->insert({{type.hash_code(), name}, resource}).second)
    {
        TF_RETURN_IF_ERROR(InsertDebugTypeName(type.hash_code(), type.name()));
        return Status::OK();
    }
    resource->Unref();
    return errors::AlreadyExists(
        "Resource ",
        container,
        "/",
        name,
        "/",
        type.name());
}

Status ResourceMgr::Cleanup(const std::string& container)
{
    {
        std::shared_lock<std::shared_mutex> l(mu_);
        if (containers_.find(container) == containers_.end())
        {
            // Nothing to cleanup.
            return Status::OK();
        }
    }
    Container* b = nullptr;
    {
        std::unique_lock<std::shared_mutex> l(mu_);
        auto iter = containers_.find(container);
        if (iter == containers_.end())
        {
            // Nothing to cleanup, it's OK (concurrent cleanup).
            return Status::OK();
        }
        b = iter->second;
        containers_.erase(iter);
    }
    CHECK(b != nullptr);
    for (const auto& p : *b)
    {
        p.second->Unref();
    }
    delete b;
    return Status::OK();
}

const tensorflow::ResourceHandleProto& HandleFromInput(
    OpKernelContext* ctx,
    int input)
{
    return ctx->input(input).base<tensorflow::ResourceHandleProto>()[0];
}

} //  end namespace tfdml
