/* Copyright (c) Microsoft Corporation.

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

#include "tfdml/core/dml_device_cache.h"

#include "tensorflow/c/logging.h"
#include "tfdml/core/dml_adapter.h"
#include "tfdml/core/dml_adapter_impl.h"
#include "tfdml/core/dml_common.h"
#include "tfdml/core/dml_device.h"
#include "tfdml/core/dml_device_state.h"
#include "tfdml/core/dml_dso_loader.h"
#include "tfdml/core/dml_util.h"

#ifdef _DEBUG
// #define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
    #define DBG_NEW new
    // Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
    // allocations to be of _CLIENT_BLOCK type
#else
    #define DBG_NEW new
#endif

using Microsoft::WRL::ComPtr;

namespace tfdml
{

// For now, only support devices that are guaranteed to work on all datatypes
// that the DML kernels use since we don't have a good way of knowing what the
// users might use the device for. In the future, we might want to use a
// different approach if the data shows that many people with old hardware
// want to use our package.
// TFDML #28244592
static bool SupportsAllDataTypes(
    const DmlAdapter& adapter,
    IDMLDevice* dml_device)
{
    std::array<DML_TENSOR_DATA_TYPE, 8> data_types = {
        DML_TENSOR_DATA_TYPE_FLOAT32,
        DML_TENSOR_DATA_TYPE_FLOAT16,
        DML_TENSOR_DATA_TYPE_UINT32,
        DML_TENSOR_DATA_TYPE_UINT16,
        DML_TENSOR_DATA_TYPE_UINT8,
        DML_TENSOR_DATA_TYPE_INT32,
        DML_TENSOR_DATA_TYPE_INT16,
        DML_TENSOR_DATA_TYPE_INT8,
    };

    return std::all_of(
        data_types.begin(),
        data_types.end(),
        [&dml_device, &adapter](DML_TENSOR_DATA_TYPE data_type)
        {
            DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT query{data_type};
            DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT support;

            auto hr = dml_device->CheckFeatureSupport(
                DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT,
                sizeof(query),
                &query,
                sizeof(support),
                &support);
            if (FAILED(hr))
            {
                TF_Log(
                    TF_WARNING,
                    "CheckFeatureSupport (data type = %d) failed for adapter: "
                    "%s",
                    static_cast<int>(data_type),
                    adapter.Name().c_str());
                return false;
            }

            return static_cast<bool>(support.IsSupported);
        });
}

// Even though we successfully created a DML Device, some APIs required by
// DirectML may still be missing. We compile a dummy identity operator to make
// sure that we don't fail later on.
static bool CanCompileOperator(IDMLDevice* dml_device)
{
    uint32_t sizes[4] = {1, 1, 1, 1};

    DML_BUFFER_TENSOR_DESC buffer_tensor_desc;
    buffer_tensor_desc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    buffer_tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
    buffer_tensor_desc.DimensionCount = 4;
    buffer_tensor_desc.Sizes = sizes;
    buffer_tensor_desc.Strides = nullptr;
    buffer_tensor_desc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
        DML_TENSOR_DATA_TYPE_FLOAT32,
        4,
        sizes,
        nullptr);
    buffer_tensor_desc.GuaranteedBaseOffsetAlignment = 0;

    DML_TENSOR_DESC tensor_desc;
    tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
    tensor_desc.Desc = &buffer_tensor_desc;

    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc;
    identity_desc.InputTensor = &tensor_desc;
    identity_desc.OutputTensor = &tensor_desc;
    identity_desc.ScaleBias = nullptr;

    DML_OPERATOR_DESC op_desc;
    op_desc.Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;
    op_desc.Desc = &identity_desc;

    ComPtr<IDMLOperator> dml_op;

    if (FAILED(dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&dml_op))))
    {
        return false;
    }

    ComPtr<IDMLCompiledOperator> dml_compiled_op;
    return SUCCEEDED(dml_device->CompileOperator(
        dml_op.Get(),
        DML_EXECUTION_FLAG_NONE,
        IID_PPV_ARGS(&dml_compiled_op)));
}

static bool SupportsDmlDevice(const DmlAdapter& adapter)
{
    D3D_FEATURE_LEVEL feature_level = adapter.IsComputeOnly()
                                          ? D3D_FEATURE_LEVEL_1_0_CORE
                                          : D3D_FEATURE_LEVEL_11_0;

    ComPtr<ID3D12Device> d3d12_device =
        TryCreateD3d12Device(adapter.Impl()->Get(), feature_level);

    if (!d3d12_device)
    {
        TF_Log(
            TF_WARNING,
            "Could not create Direct3D device for adapter: %s",
            adapter.Name().c_str());
        return false;
    }

    ComPtr<IDMLDevice> dml_device =
        TryCreateDmlDevice(d3d12_device.Get(), DML_CREATE_DEVICE_FLAG_NONE);
    if (!dml_device)
    {
        TF_Log(
            TF_WARNING,
            "Could not create DirectML device for adapter: %s",
            adapter.Name().c_str());
        return false;
    }

    return SupportsAllDataTypes(adapter, dml_device.Get()) &&
           CanCompileOperator(dml_device.Get());
}

static std::vector<DmlAdapter> FilterAdapters()
{
    // Fail early if DirectML library cannot be located or loaded.
    auto dml_handle_or = DmlCachedDsoLoader::GetDirectMLDsoHandle();
    if (!dml_handle_or.ok())
    {
        auto path = getenv("TF_DIRECTML_PATH");
        if (path)
        {
            TF_Log(
                TF_WARNING,
                "Could not load DirectML. TF_DIRECTML_PATH is set: %s",
                path);
        }
        else
        {
            TF_Log(TF_WARNING, "Could not load DirectML.");
        }

        return {};
    }

    std::vector<DmlAdapter> adapters = EnumerateAdapters();
    adapters.erase(
        std::remove_if(
            adapters.begin(),
            adapters.end(),
            [](const DmlAdapter& adapter)
            { return !SupportsDmlDevice(adapter); }),
        adapters.end());

    return adapters;
}

DmlDeviceCache& DmlDeviceCache::Instance()
{
    // Rely on magic statics to initialize this in a thread-safe manner. Note
    // that we never free this instance; it's a per-process singleton that's
    // intentionally leaked to avoid order-of-destruction issues during process
    // exit. This sounds unusual, but is done to explicitly match the behavior
    // of the CUDA device.
    static DmlDeviceCache* instance = DBG_NEW DmlDeviceCache();
    return *instance;
}

uint32_t DmlDeviceCache::GetAdapterCount() const
{
    std::unique_lock<std::mutex> lock(mutex_);

    return static_cast<uint32_t>(adapters_.size());
}

// It is a little odd that we require GPUOptions and memory_limit here, as
// those can vary per TF device instance - they're not process-global. We
// handle this by using the options and memory limit that are provided to the
// first device created on this adapter. If subsequent devices are created on
// the same adapter but with different options/memory_limit, they are ignored.
// This is unusual, but matches the behavior of the CUDA device.
const DmlDeviceState* DmlDeviceCache::GetOrCreateDeviceState(
    uint32_t adapter_index)
{
    std::unique_lock<std::mutex> lock(mutex_);

    assert(adapter_index < adapters_.size());
    assert(adapters_.size() == device_states_.size());

    if (!device_states_[adapter_index])
    {
        const DmlAdapter& adapter = adapters_[adapter_index];

        TF_Log(
            TF_INFO,
            "DirectML: creating device on adapter %u (%s)",
            adapter_index,
            adapter.Name().c_str());

        device_states_[adapter_index] = DmlDeviceState::Create(adapter);
    }

    return device_states_[adapter_index].get();
}

const DmlAdapter& DmlDeviceCache::GetAdapter(uint32_t adapter_index) const
{
    return adapters_[adapter_index];
}

DmlDeviceCache::DmlDeviceCache()
{
    std::vector<DmlAdapter> adapters = FilterAdapters();

    TF_Log(
        TF_INFO,
        "DirectML device enumeration: found %llu compatible adapters.",
        adapters.size());

    for (size_t i = 0; i < adapters.size(); ++i)
    {
        const auto& adapter = adapters[i];
        auto driver_ver = adapter.DriverVersion().parts;

        TF_VLog(1, "DirectML adapter %llu: %s", i, adapter.Name().c_str());
        TF_VLog(1, "    VendorID: %#010x", (uint32_t)adapter.VendorID());
        TF_VLog(1, "    DeviceID: %#010x", adapter.DeviceID());
        TF_VLog(
            1,
            "    Driver: %u.%u.%u.%u",
            driver_ver.a,
            driver_ver.b,
            driver_ver.c,
            driver_ver.d);
        TF_VLog(
            1,
            "    IsComputeOnly: %s",
            adapter.IsComputeOnly() ? "true" : "false");
    }

    if (adapters.size() > 1)
    {
        TF_Log(
            TF_WARNING,
            "More than one physical devices were found, but "
            "tensorflow-directml-plugin doesn't support multiple devices yet. "
            "The first available device in order of performance was selected "
            "by default. To select a different device, set the "
            "DML_VISIBLE_DEVICES environment variable to its index (e.g. "
            "DML_VISIBLE_DEVICES=1).");

        adapters_ = {adapters[0]};
    }
    else
    {
        adapters_ = std::move(adapters);
    }

    device_states_.resize(adapters_.size());
}

} // namespace tfdml