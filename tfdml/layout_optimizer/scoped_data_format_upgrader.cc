/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tfdml/layout_optimizer/scoped_data_format_upgrader.h"
#include "tfdml/layout_optimizer/transpose_context.h"

namespace tfdml
{
ScopedDataFormatUpgrader::ScopedDataFormatUpgrader(
    TransposeContext* context,
    int rank)
    : context_(context)
{
    if (rank == 5 && IsSupportedDataFormat(context_->src_format) &&
        IsSupportedDataFormat(context_->dst_format))
    {
        old_src_format_ = context_->src_format;
        old_dst_format_ = context_->dst_format;
        std::string new_src_format =
            GetUpgradedDataFormat(context_->src_format);
        std::string new_dst_format =
            GetUpgradedDataFormat(context_->dst_format);
        context_->AssignDeviceAndDataFormats(
            context_->target_device,
            new_src_format,
            new_dst_format);
        upgraded_ = true;
    }
}

ScopedDataFormatUpgrader::~ScopedDataFormatUpgrader()
{
    if (upgraded_)
    {
        context_->AssignDeviceAndDataFormats(
            context_->target_device,
            old_src_format_,
            old_dst_format_);
    }
}

bool ScopedDataFormatUpgrader::IsSupportedDataFormat(
    absl::string_view data_format)
{
    return data_format == "NHWC" || data_format == "NCHW";
}

std::string ScopedDataFormatUpgrader::GetUpgradedDataFormat(
    absl::string_view data_format)
{
    if (data_format == "NHWC")
    {
        return "NDHWC";
    }

    assert(data_format == "NCHW");
    return "NCDHW";
}
} // namespace tfdml
