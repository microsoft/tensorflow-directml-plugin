/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tfdml/runtime_adapter/mirror_pad_mode.h"

namespace tfdml
{

Status GetMirrorPaddingFromString(
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

std::string GetMirrorPadModeAttrString()
{
    return "mode: {'REFLECT', 'SYMMETRIC'}";
}

} // end namespace tfdml
