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

import math
import numpy as np
import onnx
import onnx.numpy_helper
from typing import Tuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


def extract_elem_type(elem_type: int, clip_range=None) -> Tuple[int, int, bool]:
    """
    Return Quant attribute specification based on element type and (optional)
    clipping range.
    Returns: (bitwidth, signed, is_narrow_qnt)
    """
    is_narrow = False
    # pylint: disable=no-member
    elem_map = {
        onnx.TensorProto.INT8: (8, 1),
        onnx.TensorProto.INT16: (16, 1),
        onnx.TensorProto.INT32: (32, 1),
        onnx.TensorProto.INT64: (64, 1),
        onnx.TensorProto.UINT8: (8, 0),
        onnx.TensorProto.UINT16: (16, 0),
        onnx.TensorProto.UINT32: (32, 0),
        onnx.TensorProto.UINT64: (64, 0),
    }
    if elem_type not in elem_map:
        raise ValueError("Unsupported element type: " + str(elem_type))
    (bw, is_signed) = elem_map[elem_type]
    if clip_range is not None:
        # refine bitwidth based on specified range
        (clip_min, clip_max) = clip_range
        # clip range elems must be scalars
        assert clip_min.ndim == 0 and clip_max.ndim == 0
        clip_min = clip_min.item()
        clip_max = clip_max.item()
        is_narrow = clip_min == -clip_max
        assert clip_min <= 0
        assert clip_max > 0
        n_repr = clip_max - clip_min + 1
        bw = int(math.ceil(math.log2(n_repr)))
    return (bw, is_signed, is_narrow)


# originally contributed by Keshav Gurushankar (@kgurushankar)
class QCDQToQuant(Transformation):
    """
    Fuse a chain of nodes, specifically QuantizeLinear+DequantizeLinear back
    into QONNX Quant node.
    This transform finds chains of QuantizeLinear followed by DequantizeLinear
    during the quantization process into a QONNX Quant node. If a Clip node is
    found between the QuantizeLinear+DequantizeLinear, this will be taken into
    account for the Quant bitwidth calculation.
    Input
    -----
    A model potentially quantized with QuantizeLinear, (optional) Clip and
    DequantizeLinear nodes.
    Output
    ------
    A model with QuantizeLinear, Clip and DequantizeLinear nodes re-fused back into QONNX
    Quant nodes.
    """

    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        graph = model.graph
        run_again = False
        nodes_to_remove = []
        for node in graph.node:
            if node.op_type == "DequantizeLinear":
                narrow = False
                dq_node = node
                dequant_node_index = model.get_node_index(dq_node)
                dq_inp, dq_scale, dq_zeropt = dq_node.input
                quant_candidates = model.find_direct_predecessors(dq_node)
                dq_init = model.get_initializer(dq_inp)
                dq_scale_v = model.get_initializer(dq_scale)
                dq_zeropt_v = model.get_initializer(dq_zeropt)
                if quant_candidates is None and dq_init is None:
                    continue
                if any([x is None for x in [dq_scale_v, dq_zeropt_v]]):
                    # unknown scale/zeropt for DQ, cannot continue
                    # TODO handle zeropt default according to spec
                    continue
                elif quant_candidates is None and dq_init is not None:
                    # standalone DequantizeLinear node with initializer
                    # (e.g. for weight quantization). we can treat this
                    # as if the QuantizeLinear node has already been
                    # constant-folded.
                    # read quantized weight dtype for standalone deqnt
                    q_vi = model.get_tensor_valueinfo(dq_inp)
                    (bitwidth, signed, narrow) = extract_elem_type(q_vi.type.tensor_type.elem_type)
                    # overwrite DQ initializer with scaled version
                    scaled_qnt_t = (dq_init - dq_zeropt_v) * dq_scale_v
                    scaled_qnt_t = scaled_qnt_t.astype(np.float32)
                    model.set_initializer(dq_inp, scaled_qnt_t)
                    q_inp = dq_inp
                    final_out = dq_node.output[0]
                    scale_factor, zeropt = dq_scale, dq_zeropt
                    nodes_to_remove.append(dq_node)
                elif quant_candidates[0].op_type in ["QuantizeLinear", "Clip"]:
                    clip_range = None
                    if quant_candidates[0].op_type == "Clip":
                        clip_node = quant_candidates[0]
                        clip_min = model.get_initializer(clip_node.input[1])
                        clip_max = model.get_initializer(clip_node.input[2])
                        if clip_min is None or clip_max is None:
                            # non-constant bounds for Clip, cannot convert
                            continue
                        clip_range = (clip_min, clip_max)
                        # keep following the producer chain
                        quant_candidates = model.find_direct_predecessors(clip_node)
                        if quant_candidates is None or quant_candidates[0].op_type != "QuantizeLinear":
                            # unexpected pattern, cannot convert
                            continue
                        # all good, mark Clip for removal and continue processing QuantizeLinear node
                        nodes_to_remove.append(clip_node)
                    quant_candidate = quant_candidates[0]
                    q_inp, q_scale, q_zeropt = quant_candidate.input
                    # check that zeropt/scale tensors are the same
                    q_scale_v = model.get_initializer(q_scale)
                    q_zeropt_v = model.get_initializer(q_zeropt)
                    if any([x is None for x in [q_scale_v, q_zeropt_v]]):
                        # TODO handle zeropt default
                        continue
                    qdq_v_match = (q_scale_v == dq_scale_v).all() and (q_zeropt_v == dq_zeropt_v).all()
                    qdq_nm_match = (q_scale == dq_scale) and (q_zeropt == dq_zeropt)
                    if not (qdq_nm_match or qdq_v_match):
                        continue
                    quant_node = quant_candidate
                    final_out = dq_node.output[0]
                    nodes_to_remove.append(dq_node)
                    nodes_to_remove.append(quant_node)
                    value_info = model.get_tensor_valueinfo(quant_node.output[0])
                    (bitwidth, signed, narrow) = extract_elem_type(value_info.type.tensor_type.elem_type, clip_range)
                    scale_factor, zeropt = q_scale, q_zeropt
                else:
                    # handle all other cases, skip
                    continue
                axis = get_by_name(dq_node.attribute, "axis")
                # fix scale factor for Quant (different shape expectations wrt broadcasting)
                if not (axis is None):
                    axis_i = axis.i
                    ishape = model.get_tensor_shape(dq_inp)
                    desired_shp = [1] * len(ishape)
                    desired_shp[axis_i] = dq_scale_v.shape[0]
                    dq_scale_v = dq_scale_v.reshape(desired_shp)
                    dq_zeropt_v = dq_zeropt_v.reshape(desired_shp)
                    model.set_initializer(scale_factor, dq_scale_v)
                    model.set_initializer(zeropt, dq_zeropt_v)
                # create new Quant node for suitable cases
                new_q_node_name = "Quant_" + q_inp
                bw_tensor_name = f"{new_q_node_name}_bitwidth"
                model.set_initializer(bw_tensor_name, np.asarray(bitwidth, dtype=np.float32))
                # for the zeropoint, see if we can simplify back to scalar
                if (dq_zeropt_v == 0).all():
                    model.set_initializer(zeropt, np.asarray(0, dtype=np.float32))
                fused_node = onnx.helper.make_node(
                    "Quant",
                    inputs=[
                        q_inp,
                        scale_factor,
                        zeropt,
                        bw_tensor_name,
                    ],
                    outputs=[final_out],
                    name=new_q_node_name,
                    domain="qonnx.custom_op.general",
                    narrow=1 if narrow else 0,  # depends on clip
                    rounding_mode="ROUND",  # round-to-even
                    signed=signed,
                )
                model.graph.node.insert(dequant_node_index, fused_node)
            for node_to_remove in nodes_to_remove:
                model.graph.node.remove(node_to_remove)
                run_again = True
            if run_again:
                break
        return (model, run_again)
