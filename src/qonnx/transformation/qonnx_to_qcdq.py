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
from warnings import warn

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.quant import max_int, min_int
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import MovePadAttributeToTensor, RemoveUnusedTensors


class QuantToQCDQ(Transformation):
    """Replace QONNX Quant-style quantization nodes with QuantizeLinear
    -> Clip -> DequantizeLinear (QCDQ)-style quantization nodes. The following
    restictions apply on the Quant:
    - the scale, zero-point and bitwidth inputs for Quant must be statically specified
      by an initializer
    - the bitwidth must be an integer in the range [2, 8]
    - the zero-point tensor must be zero
    - the scale must be a scalar value or 1D tensor
    - the rounding_mode attribute must be ROUND
    BipolarQuant is not (yet) supported.
    """

    def apply(self, model: ModelWrapper):
        graph = model.graph
        graph_modified = False
        nodes_to_remove = []
        for node in graph.node:
            if node.op_type == "Quant":
                # check all conditions for conversion
                inp_tname = node.input[0]
                scale_tname = node.input[1]
                zeropt_tname = node.input[2]
                bitwidth_tname = node.input[3]
                out_tname = node.output[0]
                scale_t = model.get_initializer(scale_tname)
                zeropt_t = model.get_initializer(zeropt_tname)
                bitwidth_t = model.get_initializer(bitwidth_tname)
                qnt_inst = getCustomOp(node)
                narrow = bool(qnt_inst.get_nodeattr("narrow"))
                signed = bool(qnt_inst.get_nodeattr("signed"))
                rounding_mode = qnt_inst.get_nodeattr("rounding_mode")
                last_node = node
                if zeropt_t is None or zeropt_t != 0:
                    warn("Zeropoint is undefined or nonzero, skipping " + node.name)
                    continue
                if scale_t is None or (scale_t.squeeze().ndim not in [0, 1]):
                    warn("Scale is undefined or non-0/1D, skipping " + node.name)
                    continue
                if bitwidth_tname is None or bitwidth_t.ndim != 0:
                    warn("Bitwidth is undefined or non-scalar, skipping " + node.name)
                    continue
                if bitwidth_t < 2 or bitwidth_t > 8:
                    warn("Bitwidth outside supported range [2,8], skipping " + node.name)
                    continue
                if rounding_mode != "ROUND":
                    warn("Rounding mode is not ROUND, skipping " + node.name)
                    continue
                graph_modified = True
                # QuantizeLinear wants 1D scale so squeeze
                q_scale = model.get_initializer(scale_tname)
                q_scale_new = q_scale.squeeze()
                model.set_initializer(scale_tname, q_scale_new)
                # resolve the scale factor axis
                if q_scale_new.ndim == 1:
                    qnt_axis = q_scale.shape.index(q_scale_new.shape[0])
                else:
                    qnt_axis = None
                # create new zeropoint tensor with appropriate dtype
                new_dtype = TensorProto.INT8 if signed else TensorProto.UINT8
                new_dtype_np = np.int8 if signed else np.uint8
                new_zeropt_name = model.make_new_valueinfo_name()
                new_zeropt_vi = oh.make_tensor_value_info(new_zeropt_name, new_dtype, scale_t.shape)
                new_zeropt_t = np.zeros_like(q_scale_new, dtype=new_dtype_np)
                graph.value_info.append(new_zeropt_vi)
                model.set_initializer(new_zeropt_name, new_zeropt_t)
                new_qout_name = model.make_new_valueinfo_name()
                ishape = model.get_tensor_shape(inp_tname)
                new_qout_vi = oh.make_tensor_value_info(new_qout_name, new_dtype, ishape)
                graph.value_info.append(new_qout_vi)
                n_ql = len(model.get_nodes_by_op_type("QuantizeLinear"))
                n_cl = len(model.get_nodes_by_op_type("Clip"))
                n_dql = len(model.get_nodes_by_op_type("DequantizeLinear"))
                # create the QuantizeLinear node
                new_node_qnt = oh.make_node(
                    "QuantizeLinear",
                    [inp_tname, scale_tname, new_zeropt_name],
                    [new_qout_name],
                    name="QuantizeLinear_%d" % (n_ql + 1),
                    axis=qnt_axis,
                )
                graph.node.insert(model.get_node_index(last_node) + 1, new_node_qnt)
                last_out_tname = new_qout_name
                last_node = new_node_qnt
                # determine whether clipping is needed
                # insert a Clip node if so
                bitwidth = int(bitwidth_t)
                range_min = min_int(signed, narrow, bitwidth)
                range_max = max_int(signed, narrow, bitwidth)
                if (signed and narrow) or (bitwidth < 8):
                    new_clip_oname = model.make_new_valueinfo_name()
                    new_clip_vi = oh.make_tensor_value_info(new_clip_oname, new_dtype, ishape)
                    graph.value_info.append(new_clip_vi)
                    new_clip_min = model.make_new_valueinfo_name()
                    new_clip_min_t = np.asarray(range_min, dtype=new_dtype_np)
                    # set_initializer will create the missing VIs for the
                    # clip min/max tensors here
                    model.set_initializer(new_clip_min, new_clip_min_t)
                    new_clip_max = model.make_new_valueinfo_name()
                    new_clip_max_t = np.asarray(range_max, dtype=new_dtype_np)
                    model.set_initializer(new_clip_max, new_clip_max_t)
                    new_node_clip = oh.make_node(
                        "Clip", [last_out_tname, new_clip_min, new_clip_max], [new_clip_oname], name="Clip_%d" % (n_cl + 1)
                    )
                    last_out_tname = new_clip_oname
                    graph.node.insert(model.get_node_index(last_node) + 1, new_node_clip)
                    last_node = new_node_clip
                # finally add the DequantizeLinear node
                new_node_deqnt = oh.make_node(
                    "DequantizeLinear",
                    [last_out_tname, scale_tname, new_zeropt_name],
                    [out_tname],
                    name="DequantizeLinear_%d" % (n_dql + 1),
                    axis=qnt_axis,
                )
                graph.node.insert(model.get_node_index(last_node) + 1, new_node_deqnt)
                last_node = new_node_deqnt
                nodes_to_remove.append(node)
        for node_to_remove in nodes_to_remove:
            graph.node.remove(node_to_remove)
        if graph_modified:
            # remove old tensors that are no longer used for bitwidth, zeropt etc
            model = model.transform(RemoveUnusedTensors())
            # ensure opset version is new enough for the Clip nodes we inserted
            clip_min_opset = 13
            if model.model.opset_import[0].version < clip_min_opset:
                warn(
                    "QCDQ Clip requires ONNX opset >= %d but found %d"
                    % (clip_min_opset, model.model.opset_import[0].version)
                )
                warn("Forcing opset version %s upgrade to ensure valid ONNX" % clip_min_opset)
                model.model.opset_import[0].version = clip_min_opset
                # ensure new Pad node requirements are respected
                model = model.transform(MovePadAttributeToTensor())

        return (model, graph_modified)
