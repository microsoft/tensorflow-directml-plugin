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

#include "absl/container/flat_hash_set.h"

namespace tensorflow
{
// Forward declare protos so their symbols can be removed from .so exports
class AttrValue;
class NameAttrList;
}

namespace tfdml {

// Sets *out based on the type of value.
void SetAttrValue(const absl::Span<const absl::string_view>, tensorflow::AttrValue* out);
void SetAttrValue(int64_t value, tensorflow::AttrValue* out);

}  // namespace tfdml