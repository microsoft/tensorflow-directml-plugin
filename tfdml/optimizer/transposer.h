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

#include "absl/types/span.h"
#include "tfdml/optimizer/transpose_context.h"
#include "tfdml/runtime_adapter/status.h"

namespace tensorflow
{
class NodeDef;
class AttrValue;
enum DataType : int;
} // namespace tensorflow

namespace tfdml
{
class MutationNewNode;
class MutableNodeView;

bool IsDefaultLayoutSensitiveOp(const tensorflow::NodeDef& node);
bool IsLayoutSensitiveOp(const tensorflow::NodeDef& node);
bool IsLayoutAgnosticOp(const tensorflow::NodeDef& node);

class Transposer
{
  public:
    explicit Transposer() {}

    Transposer(const Transposer&) = delete;
    Transposer& operator=(const Transposer&) = delete;

    virtual ~Transposer() {}

    // Returns true iff the node should be processed by this transposer.
    // NodeProcessors may perform additional oprand specific checks before
    // processing if necessary.
    // Following common conditions are checked:
    // * node's device matches target device
    // * node's source format matches config's source format
    // * node has output
    bool ShouldProcess(
        const TransposeContext& context,
        const MutableNodeView& node) const;

    // Transposes given node from src format to dst format. Also perform other
    // necessary operations to guarantee the graph produce the same result.
    // Eg. Add Transpose node sets before fanin ports and after fanout ports.
    virtual Status TransposeNode(
        TransposeContext* context,
        MutableNodeView* node) = 0;

    // Creates a Const node for permutation. If node with node_name already
    // exits, return and reuse it.
    Status CreateConstPermNode(
        TransposeContext* context,
        absl::string_view node_name,
        absl::string_view device,
        absl::Span<const int> permutation,
        absl::string_view control_node_name,
        MutationNewNode* added_node);

    // Creates a TransposeNode with given properties. If node with node_name
    // already exits, return and reuse it.
    // A const perm node is also created and connected to the 2nd fanin.
    // control_node_name is ignored if it is empty.
    Status CreateTransposeNode(
        TransposeContext* context,
        absl::string_view name_format,
        const tensorflow::DataType& data_type,
        absl::string_view device,
        tensorflow::TensorShapeProto fanin_shape,
        absl::Span<const int> permutation,
        absl::string_view control_node_name,
        MutationNewNode* added_node,
        std::string* transpose_node_name);

    // Update all edges between dst_node->fanin[dst_ports] and dst_node by
    // inserting an op node.
    Status UpdateFaninEdgesWithOp(
        TransposeContext* context,
        absl::Span<const int> dst_ports,
        MutableNodeView* dst_node,
        absl::string_view op);

    // Update all edges between src_node:src_ports and nodes take
    // src_node:src_ports as fanin. Also update attr _output_shape of src_node.
    Status UpdateFanoutEdgesWithOp(
        TransposeContext* context,
        absl::Span<const int> src_ports,
        MutableNodeView* src_node,
        absl::string_view op);

    // Creates a DataFromat node with given properties.
    // DataFromat op is either DataFormatVecPermute or DataFormatDimMap.
    Status CreateDataFormatNode(
        TransposeContext* context,
        absl::string_view node_name,
        absl::string_view op,
        absl::string_view device,
        const tensorflow::DataType& data_type,
        bool is_fanin_on_host,
        bool is_src_format_to_dst_format,
        MutationNewNode* added_node);

  protected:
    int GetFanoutPortRank(const MutableNodeView& node, int port) const;
    bool IsFanoutPortRankN(const MutableNodeView& node, int port, int n) const;
    bool IsFanoutPortsRankN(
        const MutableNodeView& node,
        absl::Span<const int> ports,
        int n) const;
    int GetFaninPortRank(const MutableNodeView& node, int port) const;
    bool IsFaninPortRankN(const MutableNodeView& node, int port, int n) const;

    // Checks if fanin at specified port(s) has dimensions `dims` iff fanin is a
    // Const. If fanin is not a Const, no dimensions will be checked and this
    // will return true.
    bool IsFaninPortDimsNIfConst(
        const MutableNodeView& node,
        int port,
        absl::Span<const int> dims) const;
    bool IsFaninPortsDimsNIfConst(
        const MutableNodeView& node,
        absl::Span<const int> ports,
        absl::Span<const int> dims) const;
    bool CanProcessNode(
        const TransposeContext& context,
        const MutableNodeView& node) const;
    // Update all edges between dst_node->fanin[dst_ports] and dst_node.
    // A node with op is created and inserted between all edges.
    // op is one of Transpose, DataFormatVecPermute or DataFormatDimMap.
    Status UpdateEdge(
        TransposeContext* context,
        absl::string_view name_format,
        absl::string_view op,
        const tensorflow::AttrValue* input_shape,
        bool is_in_frame,
        bool is_src_format_to_dst_format,
        const int src_port,
        const int dst_port,
        MutableNodeView* src_node,
        MutableNodeView* dst_node);
    std::string GetFaninNameFormat(
        absl::string_view node_name,
        int port,
        absl::string_view src_format,
        absl::string_view dst_format);
    std::string GetFanoutNameFormat(
        absl::string_view node_name,
        int port,
        int index,
        absl::string_view src_format,
        absl::string_view dst_format);
    std::string LayoutOptimizerNode(absl::string_view node_name);
    std::string GetReshapeNodeNameFormat(
        absl::string_view node_name,
        int index,
        absl::string_view src_format,
        absl::string_view dst_format);
    std::string GetShapeConstNodeNameFormat(
        absl::string_view node_name,
        int index);
};

class LayoutSensitiveOpTransposer : public Transposer
{
  public:
    explicit LayoutSensitiveOpTransposer() : Transposer() {}

    // Updates attrs data_format, ksize, strides of the given node to
    // dst_format. _output_shape is updated during UpdateOutputEdges.
    Status UpdateNode(TransposeContext* context, MutableNodeView* node);
};

class DefaultLayoutSensitiveOpTransposer : public LayoutSensitiveOpTransposer
{
  public:
    explicit DefaultLayoutSensitiveOpTransposer()
        : LayoutSensitiveOpTransposer()
    {
    }

    Status TransposeNode(TransposeContext* context, MutableNodeView* node)
        override;
};

} // namespace tfdml
