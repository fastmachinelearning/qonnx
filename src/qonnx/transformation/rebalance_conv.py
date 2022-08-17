# Copyright (c) 2022, Xilinx
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

import numpy as np
from onnx import TensorProto, helper

from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation


class RebalanceIm2Col(Transformation):
    """
    For certain hardware that prefers channel parallelism over feature map spatial parallelism,
    it is possible to reshape the inputs to an Im2Col node to move some of the spatial dimension
    into the channels dimension. This transformation attempts to find such Im2Col nodes, adds
    a Reshape node in front and alters their kernel/stride sizes accordingly.
    See list of conditions checked in the implementation for a full list, but one example of
    rebalancing is provided in the unit test for this transformation (test_rebalance_conv.py)
    """

    def __init__(self, extract_channels):
        super().__init__()
        self.extract_channels = int(extract_channels)

    def apply(self, model):
        graph = model.graph
        modified = False
        node_ind = 0
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                # list of conditions for rebalancing
                # TODO can the following conditions be relaxed?
                pads = inst.get_nodeattr("pad_amount")
                pads_ok = all(v == 0 for v in pads)
                strides = inst.get_nodeattr("stride")
                kernel_shape = inst.get_nodeattr("kernel_size")
                strd_k_ok = (strides == kernel_shape) and (kernel_shape[0] % self.extract_channels == 0)
                depthwise = inst.get_nodeattr("depthwise")
                group_ok = depthwise == 0
                dilations = inst.get_nodeattr("dilations")
                dilations_ok = all(v == 1 for v in dilations)
                old_ishape = model.get_tensor_shape(node.input[0])
                shape_ok = len(old_ishape) == 4
                # Im2Col uses NHWC layout
                bs = old_ishape[0]
                old_ifm = old_ishape[-1]
                chans_ok = old_ifm == 1
                if pads_ok and strd_k_ok and group_ok and dilations_ok and chans_ok and shape_ok:
                    bs, ih, iw, old_ifm = old_ishape
                    assert len(kernel_shape) == 2, "Restricted to 2D kernels "
                    assert (
                        iw % self.extract_channels == 0
                    ), """Could not
                    factor out last spatial dim %d into channels %d""" % (
                        iw,
                        self.extract_channels,
                    )
                    new_last_dim = int(iw / self.extract_channels)
                    new_ifm = self.extract_channels
                    new_ishape = (bs, ih, new_last_dim, new_ifm)
                    # TODO add a reshape node to move spatial dim into channels dim
                    running_node_index = node_ind
                    inp_reshape_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        new_ishape,
                    )
                    inp_shapedata = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.INT64,
                        [len(new_ishape)],
                    )
                    graph.value_info.append(inp_reshape_out)
                    graph.value_info.append(inp_shapedata)
                    model.set_initializer(inp_shapedata.name, np.asarray(new_ishape, dtype=np.int64))
                    inp_reshape_node = helper.make_node(
                        "Reshape", [node.input[0], inp_shapedata.name], [inp_reshape_out.name]
                    )
                    graph.node.insert(running_node_index, inp_reshape_node)
                    # rewire Im2Col input
                    node.input[0] = inp_reshape_out.name
                    # alter the attributes

                    kernel_shape[-1] = int(kernel_shape[-1] / self.extract_channels)
                    strides[-1] = int(strides[-1] / self.extract_channels)
                    inst.set_nodeattr("kernel_size", kernel_shape)
                    inst.set_nodeattr("stride", strides)
                    inst.set_nodeattr("input_shape", str(new_ishape))
                    modified = True
        return model, modified
