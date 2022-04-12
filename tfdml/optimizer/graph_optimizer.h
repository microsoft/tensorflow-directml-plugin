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

#include "tfdml/runtime_adapter/status.h"

namespace tensorflow
{
class GraphDef;
}

namespace tfdml
{
class GrapplerItem;

// An abstract interface for an algorithm for generating a candidate
// optimization of a GrapplerItem for running on a cluster.
class GraphOptimizer
{
  public:
    GraphOptimizer() {}
    virtual ~GraphOptimizer() {}

    // Routine called to allow an algorithm to propose a rewritten graph
    // for the graph, feeds and fetches in "item" to run more efficiently. If
    // the returned status is Status::OK() then *optimized_graph contains the
    // rewritten graph. Returns an error status if it failed to generate a
    // solution.
    //
    // A return value of error::Aborted() can be used signal early termination
    // of the optimizer, e.g. if the optimization turned out to be a no-op. In
    // this case the content of *optimized_graph is undefined.
    virtual Status Optimize(
        const GrapplerItem& item,
        tensorflow::GraphDef* optimized_graph) = 0;

    // Subclasses may define a version of Optimize that consumes item.
    virtual Status Optimize(
        GrapplerItem&& item,
        tensorflow::GraphDef* optimized_graph)
    {
        return Optimize(item, optimized_graph);
    }
};
} // namespace tfdml
