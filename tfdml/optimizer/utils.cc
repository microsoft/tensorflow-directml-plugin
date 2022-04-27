/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tfdml/optimizer/utils.h"

namespace tfdml {

// Returns the data type in attribute `attr_name` of `node`. If that attribute
// doesn't exist, returns DT_INVALID.
tensorflow::DataType GetDataTypeFromAttr(const tensorflow::NodeDef& node, const std::string& type_attr) {
  if (!node.attr().count(type_attr)) {
    return tensorflow::DT_INVALID;
  }
  const auto& attr = node.attr().at(type_attr);
  if (attr.value_case() != tensorflow::AttrValue::kType) {
    return tensorflow::DT_INVALID;
  }
  return attr.type();
}

}  // end namespace tfdml
