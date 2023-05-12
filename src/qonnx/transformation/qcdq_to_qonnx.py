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

import onnx
import onnx.numpy_helper
from typing import Tuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation


def extract_elem_type(elem_type: int) -> Tuple[int, int]:
    """
    Since quant nodes are being used, only int types are applicable
    Returns: (bitwidth, signed)
    """
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
    return elem_map[elem_type]


class QCDQToQuant(Transformation):
    """
    Fuse a chain of nodes, specifically QuantizeLinear+DequantizeLinear back
    into QONNX Quant node.
    This transform finds chains of QuantizeLinear followed by DequantizeLinear
    created by the Vitis AI Quantizer during the quantization process into a
    QONNX Quant node.
    Input
    -----
    A model potentially quantized by Vitis AI with QuantizeLinear and
    DequantizeLinear nodes.
    Output
    ------
    A model with QuantizeLinear and DequantizeLinear nodes re-fused back into brevitas Quant Nodes.
    """

    def __init__(self) -> None:
        super().__init__()

    def fuse(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        graph = model.graph
        for n in graph.node:
            if n.op_type == "DequantizeLinear":
                quant_candidate = model.find_direct_predecessors(n)[0]
                if quant_candidate.op_type == "QuantizeLinear":
                    self.vai_found = True
                    quant_node = quant_candidate
                    dequant_node = n

                    dequant_node_index = model.get_node_index(dequant_node)

                    model.graph.node.remove(dequant_node)
                    model.graph.node.remove(quant_node)

                    value_info = [x for x in graph.value_info if x.name == quant_node.output[0]]
                    (bitwidth, signed) = extract_elem_type(value_info[0].type.tensor_type.elem_type)

                    initializer_tensor = onnx.helper.make_tensor(
                        name=f"{n.name}_bitwidth",
                        data_type=onnx.TensorProto.FLOAT,  # pylint: disable=no-member
                        dims=(),
                        vals=[bitwidth],
                    )
                    model.graph.initializer.insert(-1, initializer_tensor)

                    # TODO import scale/zeropt tensor names properly
                    scale_factor, zeropt = "", ""

                    fused_node = onnx.helper.make_node(
                        "Quant",
                        inputs=[
                            quant_node.input[0],
                            scale_factor,
                            zeropt,
                            f"{n.name}_bitwidth",
                        ],
                        outputs=[dequant_node.output[0]],
                        name="Quant_" + str(self.quant_nodes_created),
                        domain="qonnx.custom_op.general",  # Is this correct?
                        narrow=0,  # QuantizeLinear spec says 0
                        rounding_mode="ROUND",  # QuantizeLinear spec says Round
                        signed=signed,
                    )
                    model.graph.node.insert(dequant_node_index, fused_node)
                    self.quant_nodes_created += 1
                    return (model, True)
        return (model, False)

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        model, retval = self.fuse(model)
        return (model, retval)
