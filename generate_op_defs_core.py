#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This script generates C++ structs from core TF op definitions, which are used for
registering DML pluggable device kernels. The generated structs contain a small
subset of the information in the full OpDef protobuf found here:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto

You must run this script from a python environment with the minimum target TF package
to ensure all available op definitions are generated. However, you may also hand-write
structs for custom ops only used by the DML device. The output from this script should
be redirected as follows:

python generate_op_defs_core.py > tfdml/runtime_adapter/op_defs_core.h
"""

from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.util import compat
import argparse
import os

def append_args(arg_metadata, arg_list):
    for arg in arg_list:
        if len(getattr(arg, "number_attr")) > 0:
            arg_metadata.append(f'        ArgumentDesc{{"{arg.name}", ArgumentDesc::TensorCount::SequenceAttrInt, "{arg.number_attr}"}}')
        elif len(getattr(arg, "type_list_attr")) > 0:
            arg_metadata.append(f'        ArgumentDesc{{"{arg.name}", ArgumentDesc::TensorCount::SequenceAttrList, "{arg.type_list_attr}"}}')
        else:
            arg_metadata.append(f'        ArgumentDesc{{"{arg.name}", ArgumentDesc::TensorCount::Single}}')

def append_attr(attr_metadata, attr):
    # Convert string to enum type (e.g. list(int) -> AttributeType::ListInt)
    enum_value = "AttributeType::" + attr.type.title().replace("(","").replace(")","")
    attr_metadata.append(f'        AttributeDesc{{"{attr.name}", {enum_value}}}')

def generate_op_struct(op):
    # Op names may have characters that make illegal C++ identifiers (e.g. the "Namespace>TestStringOutput" op).
    # The struct can be named anything, so long as it's unique, since it stores the original op name as a field.
    struct_name = op.name.replace(">","_")

    arg_names = []
    for arg in op.input_arg:
        arg_names.append(f'        {arg.name}')
    for arg in op.output_arg:
        arg_names.append(f'        {arg.name}')
    arg_names = ',\n'.join(arg_names)

    arg_metadata = []
    append_args(arg_metadata, op.input_arg)
    append_args(arg_metadata, op.output_arg)
    arg_metadata = ',\n'.join(arg_metadata)

    attr_names = []
    for attr in op.attr:
        # Modify attribute names that match reserved C++ keywords for the purpose of the enum
        attr_name_cpp = attr.name.replace("template","template_")
        attr_names.append(f'        {attr_name_cpp}')
    attr_names = ',\n'.join(attr_names)

    attr_metadata = []
    for attr in op.attr:
        append_attr(attr_metadata, attr)
    attr_metadata = ',\n'.join(attr_metadata)

    return f"""struct {struct_name}
{{
    static constexpr const char* name = "{op.name}";
    
    enum class Argument
    {{
{arg_names}
    }};

    static constexpr uint32_t input_arg_count = {len(op.input_arg)};
    static constexpr uint32_t output_arg_count = {len(op.output_arg)};
    static constexpr std::array<ArgumentDesc, input_arg_count + output_arg_count> argument_descs
    {{
{arg_metadata}
    }};

    enum class Attribute
    {{
{attr_names}
    }};

    static constexpr std::array<AttributeDesc, {len(op.attr)}> attribute_descs
    {{
{attr_metadata}
    }};
}};
"""

def generate_op_struct_def(op):
    # Op names may have characters that make illegal C++ identifiers (e.g. the "Namespace>TestStringOutput" op).
    # The struct can be named anything, so long as it's unique, since it stores the original op name as a field.
    struct_name = op.name.replace(">","_")
    return f"""
constexpr std::array<ArgumentDesc, {struct_name}::input_arg_count + {struct_name}::output_arg_count> {struct_name}::argument_descs;
constexpr std::array<AttributeDesc, {len(op.attr)}> {struct_name}::attribute_descs;"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_name", "-n", default="", help="Name of single op to generate (testing only)")
    args = parser.parse_args()

    buf = c_api.TF_GetAllOpList()
    data = c_api.TF_GetBuffer(buf)
    op_list = op_def_pb2.OpList()
    op_list.ParseFromString(compat.as_bytes(data))

    dir_path = os.path.dirname(os.path.abspath(__file__))
    header_path = f"{dir_path}/tfdml/runtime_adapter/op_defs_core.h"
    impl_path = f"{dir_path}/tfdml/runtime_adapter/op_defs_core.cc"

    with open(header_path, 'w') as f:
        f.write('''/* Copyright (c) Microsoft Corporation.

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

// clang-format off

// This file is generated. Do not edit it directly. See generate_op_defs_core.py.
#pragma once

namespace tfdml
{
namespace ops
{
''')

    with open(header_path, 'a') as f:
        for op in op_list.op:
            if not args.op_name or (args.op_name == op.name):
                f.write(generate_op_struct(op) + "\n")

        f.write("} // namespace tfdml\n")
        f.write("} // namespace ops\n")

    with open(impl_path, 'w') as f:
        f.write('''/* Copyright (c) Microsoft Corporation.

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

// clang-format off

// This file is generated. Do not edit it directly. See generate_op_defs_core.py.

#include "tfdml/runtime_adapter/op_defs.h"

namespace tfdml
{
namespace ops
{
''')

    with open(impl_path, 'a') as f:
        for op in op_list.op:
            if not args.op_name or (args.op_name == op.name):
                f.write(generate_op_struct_def(op) + "\n")

        f.write("} // namespace tfdml\n")
        f.write("} // namespace ops\n")