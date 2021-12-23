/* Copyright (c) Microsoft Corporation.

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "attribute.h"
#include "op_defs.h"
#include "tensorflow/c/kernels.h"
#include "tfdml/runtime_adapter/padding.h"
#include "tfdml/runtime_adapter/status.h"
#include "tfdml/runtime_adapter/tensor.h"

struct TF_OpKernelConstruction;

namespace tfdml
{
class OpKernelConstruction
{
  public:
    OpKernelConstruction(TF_OpKernelConstruction* context);

    std::string_view GetName() const
    {
        auto name = TF_OpKernelConstruction_GetName(context_);
        return {name.data, name.len};
    }

    bool HasAttr(const char* attr_name) const;

    template <typename T> Status GetAttr(const char* attr_name, T* value) const;

    template <>
    Status GetAttr<TF_DataType>(const char* attr_name, TF_DataType* value) const
    {
        CHECK(value != nullptr);
        Status status;
        TF_OpKernelConstruction_GetAttrType(
            context_,
            attr_name,
            value,
            status.raw());
        return status;
    }

    template <>
    Status GetAttr<int32_t>(const char* attr_name, int32_t* value) const
    {
        CHECK(value != nullptr);
        Status status;
        TF_OpKernelConstruction_GetAttrInt32(
            context_,
            attr_name,
            value,
            status.raw());
        return status;
    }

    template <>
    Status GetAttr<int64_t>(const char* attr_name, int64_t* value) const
    {
        CHECK(value != nullptr);
        Status status;
        TF_OpKernelConstruction_GetAttrInt64(
            context_,
            attr_name,
            value,
            status.raw());
        return status;
    }

    template <> Status GetAttr<float>(const char* attr_name, float* value) const
    {
        CHECK(value != nullptr);
        Status status;
        TF_OpKernelConstruction_GetAttrFloat(
            context_,
            attr_name,
            value,
            status.raw());
        return status;
    }

    template <> Status GetAttr<bool>(const char* attr_name, bool* value) const
    {
        CHECK(value != nullptr);
        TF_Bool tf_bool_value;
        Status status;
        TF_OpKernelConstruction_GetAttrBool(
            context_,
            attr_name,
            &tf_bool_value,
            status.raw());
        *value = tf_bool_value;
        return status;
    }

    template <>
    Status GetAttr<std::string>(const char* attr_name, std::string* value) const
    {
        CHECK(value != nullptr);
        Status attr_size_status;
        int32_t list_size;
        int32_t size_in_bytes;
        TF_OpKernelConstruction_GetAttrSize(
            context_,
            attr_name,
            &list_size,
            &size_in_bytes,
            attr_size_status.raw());

        if (!attr_size_status.ok())
        {
            return attr_size_status;
        }

        value->resize(size_in_bytes, '\0');

        Status status;
        TF_OpKernelConstruction_GetAttrString(
            context_,
            attr_name,
            value->data(),
            size_in_bytes,
            status.raw());

        return status;
    }

    template <>
    Status GetAttr<Padding>(const char* attr_name, Padding* value) const
    {
        CHECK(value != nullptr);

        std::string padding_string;
        Status status = GetAttr(attr_name, &padding_string);

        if (!status.ok())
        {
            return status;
        }

        return GetPaddingFromString(padding_string, value);
    }

    template <>
    Status GetAttr<std::vector<TF_DataType>>(
        const char* attr_name,
        std::vector<TF_DataType>* value) const
    {
        CHECK(value != nullptr);
        Status attr_size_status;
        int32_t list_size;
        int32_t size_in_bytes;
        TF_OpKernelConstruction_GetAttrSize(
            context_,
            attr_name,
            &list_size,
            &size_in_bytes,
            attr_size_status.raw());

        if (!attr_size_status.ok())
        {
            return attr_size_status;
        }

        value->resize(list_size);

        Status status;
        TF_OpKernelConstruction_GetAttrTypeList(
            context_,
            attr_name,
            value->data(),
            list_size,
            status.raw());

        return status;
    }

    template <>
    Status GetAttr<std::vector<int32_t>>(
        const char* attr_name,
        std::vector<int32_t>* value) const
    {
        CHECK(value != nullptr);
        Status attr_size_status;
        int32_t list_size;
        int32_t size_in_bytes;
        TF_OpKernelConstruction_GetAttrSize(
            context_,
            attr_name,
            &list_size,
            &size_in_bytes,
            attr_size_status.raw());

        if (!attr_size_status.ok())
        {
            return attr_size_status;
        }

        value->resize(list_size);

        Status status;
        TF_OpKernelConstruction_GetAttrInt32List(
            context_,
            attr_name,
            value->data(),
            list_size,
            status.raw());

        return status;
    }

    template <>
    Status GetAttr<std::vector<int64_t>>(
        const char* attr_name,
        std::vector<int64_t>* value) const
    {
        CHECK(value != nullptr);
        Status attr_size_status;
        int32_t list_size;
        int32_t size_in_bytes;
        TF_OpKernelConstruction_GetAttrSize(
            context_,
            attr_name,
            &list_size,
            &size_in_bytes,
            attr_size_status.raw());

        if (!attr_size_status.ok())
        {
            return attr_size_status;
        }

        value->resize(list_size);

        Status status;
        TF_OpKernelConstruction_GetAttrInt64List(
            context_,
            attr_name,
            value->data(),
            list_size,
            status.raw());

        return status;
    }

    template <>
    Status GetAttr<std::vector<float>>(
        const char* attr_name,
        std::vector<float>* value) const
    {
        CHECK(value != nullptr);
        Status attr_size_status;
        int32_t list_size;
        int32_t size_in_bytes;
        TF_OpKernelConstruction_GetAttrSize(
            context_,
            attr_name,
            &list_size,
            &size_in_bytes,
            attr_size_status.raw());

        if (!attr_size_status.ok())
        {
            return attr_size_status;
        }

        value->resize(list_size);

        Status status;
        TF_OpKernelConstruction_GetAttrFloatList(
            context_,
            attr_name,
            value->data(),
            list_size,
            status.raw());

        return status;
    }

    template <>
    Status GetAttr<std::vector<bool>>(
        const char* attr_name,
        std::vector<bool>* value) const
    {
        CHECK(value != nullptr);
        Status attr_size_status;
        int32_t list_size;
        int32_t size_in_bytes;
        TF_OpKernelConstruction_GetAttrSize(
            context_,
            attr_name,
            &list_size,
            &size_in_bytes,
            attr_size_status.raw());

        if (!attr_size_status.ok())
        {
            return attr_size_status;
        }

        value->resize(list_size);
        std::vector<TF_Bool> tf_bool_values(list_size);

        Status status;
        TF_OpKernelConstruction_GetAttrBoolList(
            context_,
            attr_name,
            tf_bool_values.data(),
            list_size,
            status.raw());

        for (int32_t i = 0; i < list_size; ++i)
        {
            (*value)[i] = tf_bool_values[i];
        }

        return status;
    }

    template <>
    Status GetAttr<std::vector<std::string>>(
        const char* attr_name,
        std::vector<std::string>* value) const
    {
        CHECK(value != nullptr);
        Status attr_size_status;
        int32_t list_size;
        int32_t size_in_bytes;
        TF_OpKernelConstruction_GetAttrSize(
            context_,
            attr_name,
            &list_size,
            &size_in_bytes,
            attr_size_status.raw());

        if (!attr_size_status.ok())
        {
            return attr_size_status;
        }

        std::vector<char*> vals(list_size);
        std::vector<size_t> lengths(list_size);
        std::vector<char> storage(size_in_bytes);

        value->resize(list_size);

        Status status;
        TF_OpKernelConstruction_GetAttrStringList(
            context_,
            attr_name,
            vals.data(),
            lengths.data(),
            list_size,
            storage.data(),
            size_in_bytes,
            status.raw());

        for (int32_t i = 0; i < list_size; ++i)
        {
            (*value)[i] = vals[i];
        }

        return status;
    }

    // Returns the number of tensors that map to the given op argument.
    Status GetArgumentTensorCount(const ArgumentDesc& arg_desc, uint32_t* value)
        const;

    AttributeValue TryGetAttributeValue(const AttributeDesc& attr_desc) const;

    void CtxFailure(const char* file, int line, const Status& s);
    void CtxFailureWithWarning(const char* file, int line, const Status& s);

  private:
    TF_OpKernelConstruction* const context_;
    Status status_;

    static Status GetPaddingFromString(
        absl::string_view str_value,
        Padding* value);
};
} // namespace tfdml
