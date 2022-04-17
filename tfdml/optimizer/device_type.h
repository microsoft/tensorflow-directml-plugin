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

#include "absl/strings/string_view.h"
#include <string>

// A DeviceType is just a string, but we wrap it up in a class to give
// some type checking as we're passing these around
class DeviceType
{
  public:
    DeviceType(const char* type);
    explicit DeviceType(absl::string_view type);
    const char* type() const;
    const std::string& type_string() const;
    bool operator<(const DeviceType& other) const;
    bool operator==(const DeviceType& other) const;
    bool operator!=(const DeviceType& other) const;

  private:
    std::string type_;
};