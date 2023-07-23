# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
# * Neither the name of qonnx nor the names of its
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

import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from typing import Tuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name

_int_op_mapping = {
    "MatMul": "MatMulInteger",
}


class MoveDequantizePastOp(Transformation):
    def __init__(self, op_types=["MatMul"]) -> None:
        super().__init__()
        self.op_types = op_types

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        run_again = False
        nodes_to_remove = []
        for node in model.graph.node:
            optype_ok = node.op_type in self.op_types
            inpcount_ok = len(node.input) == 2
            if not (optype_ok and inpcount_ok):
                continue
            op_node = node
            prod0 = model.find_producer(op_node.input[0])
            prod1 = model.find_producer(op_node.input[1])
            prod0_ok = (prod0 is not None) and (prod0.op_type == "DequantizeLinear")
            prod1_ok = (prod1 is not None) and (prod1.op_type == "DequantizeLinear")
            if not (prod0_ok and prod1_ok):
                continue
            dq0_inp_name, dq0_scale_name, dq0_zeropt_name = prod0.input
            dq1_inp_name, dq1_scale_name, dq1_zeropt_name = prod1.input
            dq0_scale = model.get_initializer(dq0_scale_name)
            dq1_scale = model.get_initializer(dq1_scale_name)
            # scale factors must be static
            if (dq0_scale is None) or (dq1_scale is None):
                continue
            # scale factors must be equal
            # TODO support non-equal scale factor case
            if not np.isclose(dq0_scale, dq1_scale).all():
                continue
            dq0_zeropt = model.get_initializer(dq0_zeropt_name)
            dq1_zeropt = model.get_initializer(dq1_zeropt_name)
            # zeropoints must be static
            if (dq0_zeropt is None) or (dq1_zeropt is None):
                continue
            # zeropoints must be equal
            if not np.isclose(dq0_zeropt, dq1_zeropt).all():
                continue
            # zeropoints must be zero
            # TODO support non-0 zeropoint
            if not (dq0_zeropt == 0).all():
                continue
            # all good! proceed with re-wiring
            # create a new result tensor in int32
            new_res_name = model.make_new_valueinfo_name()
            res_shape = model.get_tensor_shape(op_node.output[0])
            res_dtype = TensorProto.INT32
            model.set_tensor_shape(new_res_name, res_shape, res_dtype)
            orig_output_name = op_node.output[0]
            # change op type to integer equiv
            op_node.op_type = _int_op_mapping[op_node.op_type]
            # wire inputs of original DQ nodes directly into
            # this node
            op_node.input[0] = dq0_inp_name
            op_node.input[1] = dq1_inp_name
            op_node.output[0] = new_res_name
            qnt_axis = get_by_name(prod0.attribute, "axis")
            # create a new DQ node for output
            new_node_deqnt = oh.make_node(
                "DequantizeLinear",
                [new_res_name, dq0_scale_name],
                [orig_output_name],
                name="DequantizeLinear_%s" % (op_node.name),
                axis=qnt_axis,
            )
            model.graph.node.insert(model.get_node_index(op_node) + 1, new_node_deqnt)
            # mark original DQ nodes for removal
            nodes_to_remove += [prod0, prod1]
            run_again = True
        for node_to_remove in nodes_to_remove:
            model.graph.node.remove(node_to_remove)
        return (model, run_again)
