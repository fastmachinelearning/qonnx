# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

import qonnx.core.onnx_exec as oxe
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes


class FoldConstantsFiltered(Transformation):
    """Replace the output of a node with const-only inputs with a precomputed
    result. Use the match_filter_fxn(model, node) function to decide which nodes
    can be eligible for const folding."""

    def __init__(self, match_filter_fxn):
        super().__init__()
        self.match_filter_fxn = match_filter_fxn

    def apply(self, model):
        opset_version = model.model.opset_import[0].version
        graph = model.graph
        node_ind = 0
        graph_modified = False
        execution_context = model.make_empty_exec_context()
        nodes_to_remove = []
        for n in graph.node:
            node_ind += 1
            node_inp_inits = list(map(lambda x: model.get_initializer(x), n.input))
            node_inp_dyn = list(filter(lambda x: x is None, node_inp_inits))
            node_out = n.output[0]
            is_all_constant_inputs = len(node_inp_dyn) == 0
            ishape = model.get_tensor_shape(n.input[0])
            is_const_shape = (n.op_type == "Shape") and (ishape is not None)
            eligible = self.match_filter_fxn(model, n)
            if (is_all_constant_inputs or is_const_shape) and eligible:
                # this node has no dynamic inputs, only constant ones -- so we can
                # do constant folding.
                oxe.execute_node(n, execution_context, graph, opset_version=opset_version)
                # use the execution result as an initializer
                model.set_initializer(node_out, execution_context[node_out])
                # remove old node
                nodes_to_remove.append(n)
                graph_modified = True
        for node in nodes_to_remove:
            model.graph.node.remove(node)
        if graph_modified:
            model = model.transform(InferShapes())
        return (model, graph_modified)


class FoldConstants(Transformation):
    """Replace the output of a node with const-only inputs with a precomputed
    result. Skip any op types given in exclude_op_types."""

    def __init__(self, exclude_op_types=["Quant", "BipolarQuant"]):
        super().__init__()
        self.exclude_op_types = exclude_op_types

    def apply(self, model):
        opset_version = model.model.opset_import[0].version
        node_ind = 0
        graph_modified = False
        execution_context = model.make_empty_exec_context()
        nodes_to_remove = []
        for n in model.graph.node:
            node_ind += 1
            node_inp_inits = list(map(lambda x: model.get_initializer(x), n.input))
            node_inp_dyn = list(filter(lambda x: x is None, node_inp_inits))
            node_out = n.output[0]
            is_all_constant_inputs = len(node_inp_dyn) == 0
            if len(n.input) > 0:
                ishape = model.get_tensor_shape(n.input[0])
                is_const_shape = (n.op_type == "Shape") and (ishape is not None)
                is_no_input = False
            else:
                is_no_input = True
            exclude = n.op_type in self.exclude_op_types
            if (is_all_constant_inputs or is_const_shape or is_no_input) and not exclude:
                # this node has no (dynamic) inputs, only constant ones -- so we can
                # do constant folding. to ensure any missing ValueInfos from initializers
                # are populated, we 'touch' the shape of all inputs first below.
                for inp in n.input:
                    model.get_tensor_shape(inp, fix_missing_init_shape=True)
                oxe.execute_node(n, execution_context, model.graph, opset_version=opset_version)
                # use the execution result as an initializer
                model.set_initializer(node_out, execution_context[node_out])
                # remove old node
                nodes_to_remove.append(n)
                graph_modified = True
                # Exit the loop here, after changing a single node. The
                # ModelWrapper ensures to repeat this transformatin as long as
                # there are suitable nodes available.
                # See https://github.com/fastmachinelearning/qonnx/issues/104
                break
        for node in nodes_to_remove:
            model.graph.node.remove(node)
        if graph_modified:
            model = model.transform(InferShapes())
        return (model, graph_modified)
