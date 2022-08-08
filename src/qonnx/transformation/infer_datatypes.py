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

import qonnx.custom_op.registry as registry
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name, is_finn_op


def _infer_node_datatype(model, node):
    """Infer output datatype(s) for a particular node. Returns True if any
    changes were made."""
    dt_identity_optypes = [
        "Reshape",
        "Transpose",
        "Flatten",
        "Slice",
        "Gather",
        "GatherElements",
        "GatherND",
        "Identity",
        "Expand",
        "Flatten",
        "MaxPool",
        "GlobalMaxPool",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "Squeeze",
        "Unsqueeze",
        "Tile",
    ]
    idtypes = list(map(lambda x: model.get_tensor_datatype(x), node.input))
    odtypes = list(map(lambda x: model.get_tensor_datatype(x), node.output))
    op_type = node.op_type
    if is_finn_op(node.domain):
        # handle DataType inference for CustomOp
        try:
            # lookup op_type in registry of CustomOps
            inst = registry.getCustomOp(node)
            inst.infer_node_datatype(model)
        except KeyError:
            # exception if op_type is not supported
            raise Exception("Custom op_type %s is currently not supported." % op_type)
    else:
        if node.op_type == "Sign":
            # always produces bipolar outputs
            model.set_tensor_datatype(node.output[0], DataType["BIPOLAR"])
        elif node.op_type in ["MatMul", "Conv"]:
            if len(list(filter(lambda x: x == DataType["FLOAT32"], idtypes))) != 0:
                # node has at least one float input, output is also float
                model.set_tensor_datatype(node.output[0], DataType["FLOAT32"])
            else:
                # TODO compute minimum / maximum result to minimize bitwidth
                # use (u)int32 accumulators for now
                has_signed_inp = len(list(filter(lambda x: x.signed(), idtypes))) != 0
                if has_signed_inp:
                    odtype = DataType["INT32"]
                else:
                    odtype = DataType["UINT32"]
                model.set_tensor_datatype(node.output[0], odtype)
        elif node.op_type in ["Resize", "Upsample"]:
            mode = get_by_name(node.attribute, "mode").s
            if mode is None:
                mode = "nearest"
            else:
                mode = mode.decode("UTF-8")
            if mode == "nearest":
                # set output dtype = input dtype
                idtype = model.get_tensor_datatype(node.input[0])
                model.set_tensor_datatype(node.output[0], idtype)
        elif node.op_type in ["Add", "Sub"]:
            if len(list(filter(lambda x: not (x.is_integer() or x.is_fixed_point()), idtypes))) != 0:
                # node has at least one non-quantized input, output is also float
                model.set_tensor_datatype(node.output[0], DataType["FLOAT32"])
            else:
                has_signed_inp = len(list(filter(lambda x: x.signed(), idtypes))) != 0
                if has_signed_inp or (node.op_type == "Sub"):
                    odtype = DataType["INT32"]
                else:
                    odtype = DataType["UINT32"]
                model.set_tensor_datatype(node.output[0], odtype)
        elif node.op_type in dt_identity_optypes:
            # set output dtype = input dtype
            idtype = model.get_tensor_datatype(node.input[0])
            model.set_tensor_datatype(node.output[0], idtype)
        else:
            # unknown, assume node produces float32 outputs
            for o in node.output:
                # check if output datatype is already set to a value != FLOAT32
                odtype = model.get_tensor_datatype(o)
                if odtype is not None and odtype != DataType["FLOAT32"]:
                    # don't change data type
                    model.set_tensor_datatype(o, odtype)
                else:
                    model.set_tensor_datatype(o, DataType["FLOAT32"])
    # compare old and new output dtypes to see if anything changed
    new_odtypes = list(map(lambda x: model.get_tensor_datatype(x), node.output))
    graph_modified = new_odtypes != odtypes
    return graph_modified


class InferDataTypes(Transformation):
    """Infer QONNX DataType info for all intermediate/output tensors based on
    inputs and node type."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for node in graph.node:
            graph_modified |= _infer_node_datatype(model, node)
        return (model, graph_modified)
