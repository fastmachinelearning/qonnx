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
from qonnx.core.datatype import DataType, ScaledIntType
from qonnx.transformation.base import Transformation
from qonnx.transformation.qcdq_to_qonnx import extract_elem_type
from qonnx.util.basic import get_by_name, is_finn_op


def is_scaled_int(x):
    # can treat both integer, fixed point and scaled int as scaled int
    return x.is_integer() or x.is_fixed_point() or isinstance(x, ScaledIntType)


def infer_mac_result_dtype(idtypes, odtype_orig, possible_negation):
    # will default to original output dtype unless specific cases detected
    ret = odtype_orig
    # result may be signed if:
    # - any of the operands are signed
    # - the operator itself may induce negation (like subtraction)
    maybe_signed = possible_negation or any([x.signed() for x in idtypes])
    if all([x.is_integer() for x in idtypes]):
        ret = DataType["INT32"] if maybe_signed else DataType["UINT32"]
    elif all([is_scaled_int(x) for x in idtypes]):
        ret = DataType["SCALEDINT<32>"]
    return ret


def _infer_node_datatype(model, node, allow_scaledint_dtypes):
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
        "Pad",
        "Concat",
        "Clip",
    ]
    mac_like_optypes = ["MatMul", "Gemm", "Conv", "Add", "Sub", "Mul"]
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
        elif node.op_type in mac_like_optypes:
            possible_negation = node.op_type in ["Sub"]
            odtype_orig = model.get_tensor_datatype(node.output[0])
            odtype = infer_mac_result_dtype(idtypes, odtype_orig, possible_negation=possible_negation)
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
        elif node.op_type in dt_identity_optypes:
            # set output dtype = input dtype
            idtype = model.get_tensor_datatype(node.input[0])
            model.set_tensor_datatype(node.output[0], idtype)
        elif node.op_type == "QuantizeLinear":
            # retrieve from output tensor dtype
            ovi = model.get_tensor_valueinfo(node.output[0])
            (bitwidth, signed, _) = extract_elem_type(ovi.type.tensor_type.elem_type)
            prefix = "INT" if signed else "UINT"
            ret = DataType["%s%d" % (prefix, bitwidth)]
            model.set_tensor_datatype(node.output[0], ret)
        elif node.op_type == "DequantizeLinear":
            # retrieve from input tensor dtype
            ivi = model.get_tensor_valueinfo(node.input[0])
            (bitwidth, signed, _) = extract_elem_type(ivi.type.tensor_type.elem_type)
            ret = DataType["SCALEDINT<%d>" % (bitwidth)]
            model.set_tensor_datatype(node.output[0], ret)
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
    # if scaled-int dtype inference is disabled, replace those with FLOAT32
    if not allow_scaledint_dtypes:
        for out in node.output:
            if "SCALEDINT" in model.get_tensor_datatype(out).get_canonical_name():
                model.set_tensor_datatype(out, DataType["FLOAT32"])
    # compare old and new output dtypes to see if anything changed
    new_odtypes = list(map(lambda x: model.get_tensor_datatype(x), node.output))
    graph_modified = new_odtypes != odtypes
    return graph_modified


class InferDataTypes(Transformation):
    """Infer QONNX DataType info for all intermediate/output tensors based on
    inputs and node type."""

    def __init__(self, allow_scaledint_dtypes=False):
        super().__init__()
        self.allow_scaledint_dtypes = allow_scaledint_dtypes

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for node in graph.node:
            graph_modified |= _infer_node_datatype(model, node, self.allow_scaledint_dtypes)
        return (model, graph_modified)
