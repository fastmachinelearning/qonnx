# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of QONNX nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy
import pathlib
import tempfile
from onnx import helper

from qonnx.transformation.base import Transformation


class PartitionFromLambda(Transformation):
    """Split a graph into partitions. Each resulting partition node has a model
    attribute indicating the path to the subordinate onnx file.
    Cleanup and InferShapes() transformations should be applied first.

    Argument 0: partitioning
    * Function performing the mapping: node -> partition_id (int or string)
    * Partitions may not cover the graph completely (nodes mapped to -1 are retained)
    * Mapping must return -1 for GenericPartition nodes

    Argument 1 (optional): partition_dir
    * Manually define where to save the partition models
    """

    def __init__(self, partitioning=lambda node: -1, partition_dir=None):
        super().__init__()
        self.partitioning = partitioning
        self.partition_dir = partition_dir

    def apply(self, model):
        # identify partitions to create
        original_nodes = list(model.graph.node)
        partition_ids = set(list(map(self.partitioning, original_nodes)))
        partition_ids.discard(-1)

        # prepare dir for generated .onnx models
        if self.partition_dir is None:
            self.partition_dir = tempfile.mkdtemp(prefix="partitioning_")
        else:
            pathlib.Path(self.partition_dir).mkdir(parents=True, exist_ok=True)

        for partition_id in partition_ids:
            all_nodes = list(model.graph.node)
            partition_nodes = list(filter(lambda x: self.partitioning(x) == partition_id, all_nodes))
            non_partition_nodes = list(filter(lambda x: x not in partition_nodes, all_nodes))

            # partition the model into two models
            p_model = copy.deepcopy(model)
            non_p_model = model
            # remove all non-partition nodes from the partition model
            for node_to_remove in non_partition_nodes:
                p_model.graph.node.remove(node_to_remove)

            # identify the entry and exit points for the partition part
            p_in = []
            p_out = []
            p_start_ind = 0
            for node in p_model.graph.node:
                for in_tensor in node.input:
                    # check if producer has been removed = lies outside the partition
                    has_initializer = in_tensor in [x.name for x in p_model.graph.initializer]
                    has_producer = p_model.find_producer(in_tensor) is not None
                    if not has_initializer and not has_producer:
                        # the same tensor could feed multiple nodes within the partition
                        # (e.g. for residual connections), so we avoid duplicates
                        if in_tensor not in p_in:
                            p_in.append(in_tensor)
                        # keep track of where this partition starts topologically
                        if p_start_ind == 0:
                            p_start_ind = all_nodes.index(node)
                for out_tensor in node.output:
                    # check if tensor is top-level output
                    # or has a consumer outside the partition
                    if out_tensor in [x.name for x in model.graph.output]:
                        if out_tensor not in p_out:
                            p_out.append(out_tensor)
                    else:
                        for consumer in model.find_consumers(out_tensor):
                            if self.partitioning(consumer) != partition_id:
                                if out_tensor not in p_out:
                                    p_out.append(out_tensor)

            p_in_vi = list(map(lambda x: p_model.get_tensor_valueinfo(x), p_in))
            p_out_vi = list(map(lambda x: p_model.get_tensor_valueinfo(x), p_out))

            # check if partitioning is legal (i.e. creates no cycles)
            to_check = [model.find_producer(x) for x in p_in]
            while len(to_check) > 0:
                next_to_check = []
                for node in to_check:
                    if node is not None:
                        assert (
                            self.partitioning(node) != partition_id
                        ), """cycle-free graph violated: partition depends on itself"""
                        # print(node)
                        predecessors = model.find_direct_predecessors(node)
                        if predecessors is not None:
                            next_to_check.extend(predecessors)
                to_check = next_to_check

            # set p graph in/out to be p_in/p_out
            while len(p_model.graph.input) > 0:
                p_model.graph.input.pop()
            for i in p_in_vi:
                p_model.graph.input.append(i)

            while len(p_model.graph.output) > 0:
                p_model.graph.output.pop()
            for o in p_out_vi:
                p_model.graph.output.append(o)

            # remove redundant input and output value_info entries
            for i in p_in_vi:
                if i in p_model.graph.value_info:
                    p_model.graph.value_info.remove(i)

            for o in p_out_vi:
                if o in p_model.graph.value_info:
                    p_model.graph.value_info.remove(o)

            # save partition model
            p_model_filename = self.partition_dir + "/partition_" + str(partition_id) + ".onnx"
            p_model.cleanup()
            p_model.save(p_model_filename)

            # insert GenericPartition node
            p_node = helper.make_node(
                "GenericPartition",
                p_in,
                p_out,
                name="GenericPartition_" + str(partition_id),
                # use the model attribute to mark the partition model
                model=p_model_filename,
                domain="qonnx.custom_op.general",
            )
            non_p_model.graph.node.insert(p_start_ind, p_node)

            # remove all partition nodes from the parent model
            # do this after inserting the p_node for easier p_start_ind handling
            for node_to_remove in partition_nodes:
                non_p_model.graph.node.remove(node_to_remove)

            model = non_p_model

        return (model, False)


class PartitionFromDict(Transformation):
    """Split a graph into partitions. Each resulting partition node has a model
    attribute indicating the path to the subordinate onnx file.
    Cleanup and InferShapes() transformations should be applied first.

    This transformation builds on PartitionFromLambda() and takes a dictionary that
    defines partitions based on node indices.

    Argument 0: partitioning
    * Dictionary with the following format:  { partition_id : node_index_list }
    * Example: {0 : [3,4,5], 1 : range(10, 15)}

    Argument 1 (optional): partition_dir
    * Manually define where to save the partition models
    """

    def __init__(self, partitioning={}, partition_dir=None):
        super().__init__()
        self.partitioning = partitioning
        self.partition_dir = partition_dir

    def apply(self, model):
        # prepare node -> int assignment fct.
        def partitioning_func(node):
            if node not in model.graph.node:
                return -1
            node_index = list(model.graph.node).index(node)
            candidates = list(filter(lambda key_value: node_index in key_value[1], self.partitioning.items()))
            if len(candidates) == 0:
                return -1
            assert len(candidates) == 1, f"single node assigned to multiple partitions: {candidates}"
            return candidates[0][0]  # partition_id

        # apply partitioning
        model = model.transform(PartitionFromLambda(partitioning_func, self.partition_dir))
        return (model, False)
