# Copyright (c) 2021 Xilinx, Inc.
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

from onnx import helper as oh

import qonnx.core.data_layout as data_layout
from qonnx.transformation.base import Transformation


class MakeInputChannelsLast(Transformation):
    """For networks with an input using the NCx data layout, add a transpose node
    at the beginning and mark the input as using NxC (channels-last)."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph_in_name = model.graph.input[0].name
        graph_new_in_name = graph_in_name + "_transposed"
        orig_ishape = model.get_tensor_shape(graph_in_name)
        ndim = len(orig_ishape)
        if ndim == 2:
            # assume NC layout, no action needed
            return (model, False)
        elif ndim > 2:
            orig_layout = model.get_tensor_layout(graph_in_name)
            if orig_layout == data_layout.get_channels_last_layout_for_ndims(ndim):
                # already marked as channels-last, no action needed
                return (model, False)
            else:
                # determine channels-last shape and required permutation to
                # go from channels-last to previous format
                new_perm = list(range(ndim))
                new_perm.remove(ndim - 1)
                new_perm.insert(1, ndim - 1)
                new_ishape = list(orig_ishape)
                new_ishape.remove(orig_ishape[1])
                new_ishape.append(orig_ishape[1])
                # create and insert transpose node
                t_trans_node = oh.make_node("Transpose", [graph_in_name], [graph_new_in_name], perm=new_perm)
                model.graph.node.insert(0, t_trans_node)
                # rewire all consumers of original input to transpose's output
                consumers = model.find_consumers(graph_in_name)
                for cons in consumers:
                    if cons == t_trans_node:
                        continue
                    for i, ci in enumerate(cons.input):
                        if ci == graph_in_name:
                            cons.input[i] = graph_new_in_name
                # set tensor shapes and layouts
                model.set_tensor_shape(graph_in_name, new_ishape)
                model.set_tensor_shape(graph_new_in_name, orig_ishape)
                model.set_tensor_layout(graph_in_name, data_layout.get_channels_last_layout_for_ndims(ndim))
                model.set_tensor_layout(
                    graph_new_in_name,
                    data_layout.get_channels_first_layout_for_ndims(ndim),
                )
                # single iteration is enough so return model_was_changed=False
                return (model, False)
