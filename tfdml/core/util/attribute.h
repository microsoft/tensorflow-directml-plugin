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

#pragma once

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"

using PrimitiveAttribute =
    absl::variant<int32_t, int64_t, float, bool, std::string>;

using Attribute = absl::variant<PrimitiveAttribute,
                                absl::InlinedVector<PrimitiveAttribute, 4>>;

using NameAttributePair = std::pair<std::string, Attribute>;

// TODO: Remove this when/if the following PR gets merged
// https://github.com/tensorflow/tensorflow/pull/52157
struct BaseAttributes {
  virtual ~BaseAttributes() = default;
  virtual absl::Span<const NameAttributePair> GetNamedAttributes() const = 0;
};
