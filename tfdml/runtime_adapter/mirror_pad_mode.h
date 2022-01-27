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

#pragma once

#include "tfdml/runtime_adapter/status.h"
#include <string>

namespace tfdml
{

// REFLECT: Border elements are not mirrored to padded regions.
// SYMMETRIC: Border elements are mirrored to padded regions.
//
// For example, if two elements are padded to the right of an array [1, 2, 3],
// then the result is [1, 2, 3, 2, 1] for REFLECT mode, and is [1, 2, 3, 3, 2]
// for SYMMETRIC mode.
enum MirrorPadMode
{
    REFLECT = 1,
    SYMMETRIC = 2,
};

// Return the string containing the list of valid padding modes, that can be
// used as an Attr() in REGISTER_OP.
std::string GetMirrorPadModeAttrString();

// Sets mirror pad mode based on the given string padding value.
Status GetMirrorPaddingFromString(
    absl::string_view str_value,
    MirrorPadMode* value);

} // end namespace tfdml
