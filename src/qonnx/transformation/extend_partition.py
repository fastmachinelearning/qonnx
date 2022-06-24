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

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.util.basic import get_by_name


class ExtendPartition(Transformation):
    """Extends GenericPartition type nodes by inserting the graph pointed to by
    the model attribute.
    Argument 0: extend_index
    * List that contains the node indices of the GenericPartition nodes
    """

    def __init__(self, extend_index):
        super().__init__()
        self.extend_index = extend_index

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        partition_nodes_dict = {ind: n for ind, n in enumerate(graph.node) if n.op_type == "GenericPartition"}

        for k, v in partition_nodes_dict.items():
            if k in self.extend_index:
                path_to_model = get_by_name(v.attribute, "model", "name").s.decode("utf-8")
                model_partition = ModelWrapper(path_to_model)

                # Append nodes
                for partition_node in model_partition.graph.node:
                    graph.node.append(partition_node)

                # Append value infos
                partition_valueinfos = [x.name for x in model_partition.graph.value_info]
                for vi_name in partition_valueinfos:
                    vi = model_partition.get_tensor_valueinfo(vi_name)
                    graph.value_info.append(vi)

                # Append initializers
                partition_initializers = [x for x in model_partition.graph.initializer]
                for i in partition_initializers:
                    graph.initializer.append(i)

                # Append tensor annotations, except for the input/output tensors
                # of the partitioned graph, as these will be present in the
                # 'upper' model.
                in_out_names = [x.name for x in model_partition.graph.input]
                in_out_names += [x.name for x in model_partition.graph.output]
                partition_annotations = [
                    x for x in model_partition.graph.quantization_annotation if x.tensor_name not in in_out_names
                ]
                for a in partition_annotations:
                    graph.quantization_annotation.append(a)

                graph.node.remove(v)
                graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph())

        return (model, graph_modified)
