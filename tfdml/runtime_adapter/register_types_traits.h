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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tfdml {

// Remap POD types by size to equivalent proxy types. This works
// since all we are doing is copying data around.
struct UnusableProxyType;
template <typename Device, int size>
struct proxy_type_pod {
  typedef UnusableProxyType type;
};
template <>
struct proxy_type_pod<Eigen::ThreadPoolDevice, 8> {
  typedef ::int64_t type;
};
template <>
struct proxy_type_pod<Eigen::ThreadPoolDevice, 4> {
  typedef int32_t type;
};
template <>
struct proxy_type_pod<Eigen::ThreadPoolDevice, 2> {
  typedef int16_t type;
};
template <>
struct proxy_type_pod<Eigen::ThreadPoolDevice, 1> {
  typedef int8_t type;
};


/// If POD we use proxy_type_pod, otherwise this maps to identity.
template <typename Device, typename T>
struct proxy_type {
  typedef typename std::conditional<
      std::is_arithmetic<T>::value,
      typename proxy_type_pod<Device, sizeof(T)>::type, T>::type type;
  static_assert(sizeof(type) == sizeof(T), "proxy_type_pod is not valid");
};

}  // namespace tfdml
