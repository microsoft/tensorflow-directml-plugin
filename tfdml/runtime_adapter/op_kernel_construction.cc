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

#include "op_kernel_construction.h"
#include "tensorflow/c/kernels.h"
#include "tfdml/runtime_adapter/padding.h"

namespace tfdml
{
OpKernelConstruction::OpKernelConstruction(TF_OpKernelConstruction* context)
    : context_(context)
{
}

void OpKernelConstruction::CtxFailure(
    const char* file,
    int line,
    const Status& s)
{
    TF_VLog(
        1,
        "OP_REQUIRES failed at %s:%d : %s",
        file,
        line,
        s.error_message());
    status_.Update(s);
    TF_OpKernelConstruction_Failure(context_, status_.raw());
}

void OpKernelConstruction::CtxFailureWithWarning(
    const char* file,
    int line,
    const Status& s)
{
    TF_Log(
        TF_WARNING,
        "OP_REQUIRES failed at %s:%d : %s",
        file,
        line,
        s.error_message());
    status_.Update(s);
    TF_OpKernelConstruction_Failure(context_, status_.raw());
}

Status OpKernelConstruction::GetArgumentTensorCount(
    const ArgumentDesc& arg_desc,
    uint32_t* tensor_count) const
{
    CHECK(tensor_count != nullptr);

    switch (arg_desc.tensor_count)
    {
    case ArgumentDesc::TensorCount::Single: {
        *tensor_count = 1;
        return Status::OK();
    }

    case ArgumentDesc::TensorCount::SequenceAttrInt: {
        int32_t value = 0;
        auto status = GetAttr<int32_t>(arg_desc.sequence_attr_name, &value);
        if (status.ok())
        {
            *tensor_count = static_cast<uint32_t>(value);
        }
        return status;
    }

    case ArgumentDesc::TensorCount::SequenceAttrList: {
        Status status;
        int32_t list_size;
        int32_t size_in_bytes;
        TF_OpKernelConstruction_GetAttrSize(
            context_,
            arg_desc.sequence_attr_name,
            &list_size,
            &size_in_bytes,
            status.raw());
        if (status.ok())
        {
            *tensor_count = static_cast<uint32_t>(list_size);
        }
        return status;
    }
    default: CHECK(false);
    }
}

template <typename T>
AttributeValue TryGetValue(
    const OpKernelConstruction& ctx,
    const char* attr_name)
{
    T value;
    auto status = ctx.GetAttr<T>(attr_name, &value);
    if (!status.ok())
    {
        return absl::nullopt;
    }
    return value;
}

AttributeValue OpKernelConstruction::TryGetAttributeValue(
    const AttributeDesc& attr_desc) const
{
    switch (attr_desc.type)
    {
    case AttributeType::Type:
        return TryGetValue<TF_DataType>(*this, attr_desc.name);
    case AttributeType::Int: return TryGetValue<int64_t>(*this, attr_desc.name);
    case AttributeType::Float: return TryGetValue<float>(*this, attr_desc.name);
    case AttributeType::Bool: return TryGetValue<bool>(*this, attr_desc.name);
    case AttributeType::String:
        return TryGetValue<std::string>(*this, attr_desc.name);
    case AttributeType::ListType:
        return TryGetValue<std::vector<TF_DataType>>(*this, attr_desc.name);
    case AttributeType::ListInt:
        return TryGetValue<std::vector<int64_t>>(*this, attr_desc.name);
    case AttributeType::ListFloat:
        return TryGetValue<std::vector<float>>(*this, attr_desc.name);
    case AttributeType::ListBool:
        return TryGetValue<std::vector<bool>>(*this, attr_desc.name);
    case AttributeType::ListString:
        return TryGetValue<std::vector<std::string>>(*this, attr_desc.name);
        // These attribute types cannot be retrieved with the C API
        // (#36968411):
    case AttributeType::Shape:
    case AttributeType::Func:
    case AttributeType::Tensor:
    case AttributeType::ListShape:
    case AttributeType::ListFunc:
    case AttributeType::ListTensor:
    default: return absl::nullopt;
    }
}

bool OpKernelConstruction::HasAttr(const char* attr_name) const
{
    Status status;
    bool hasAttr =
        TF_OpKernelConstruction_HasAttr(context_, attr_name, status.raw());
    CHECK(status.ok());
    return hasAttr;
}

Status OpKernelConstruction::GetPaddingFromString(
    absl::string_view str_value,
    Padding* value)
{
    if (str_value == "SAME")
    {
        *value = SAME;
    }
    else if (str_value == "VALID")
    {
        *value = VALID;
    }
    else if (str_value == "EXPLICIT")
    {
        *value = EXPLICIT;
    }
    else
    {
        return errors::NotFound(str_value, " is not an allowed padding type");
    }
    return Status::OK();
}

Status OpKernelConstruction::GetMirrorPaddingFromString(
    absl::string_view str_value,
    MirrorPadMode* value)
{
    if (str_value == "REFLECT")
    {
        *value = REFLECT;
    }
    else if (str_value == "SYMMETRIC")
    {
        *value = SYMMETRIC;
    }
    else
    {
        return errors::NotFound(str_value, " is not an allowed padding type");
    }
    return Status::OK();
}

} // namespace tfdml
