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

#pragma once

#include "absl/strings/string_view.h"

namespace tfdml
{
class TransposeContext;

class ScopedDataFormatUpgrader
{
  public:
    ScopedDataFormatUpgrader(TransposeContext* context, int rank);
    ScopedDataFormatUpgrader(const ScopedDataFormatUpgrader&) = delete;
    ScopedDataFormatUpgrader& operator=(const ScopedDataFormatUpgrader&) =
        delete;
    ~ScopedDataFormatUpgrader();

  private:
    bool IsSupportedDataFormat(absl::string_view data_format);
    std::string GetUpgradedDataFormat(absl::string_view data_format);
    TransposeContext* context_ = nullptr;
    bool upgraded_ = false;
    std::string old_src_format_;
    std::string old_dst_format_;
};
} // namespace tfdml