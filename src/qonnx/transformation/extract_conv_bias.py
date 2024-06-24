# Copyright (c) 2021, Xilinx
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

import warnings
from onnx import helper

from qonnx.transformation.base import Transformation


class ExtractBiasFromConv(Transformation):
    def apply(self, model):
        graph = model.graph
        node_ind = 0
        nodes_to_remove = []
        for n in graph.node:
            node_ind += 1
            if n.op_type in ["Conv", "ConvTranspose"]:
                # Check if the node has a bias input
                if len(n.input) > 2:
                    # Extract bias
                    bias = model.get_initializer(n.input[2])
                    if bias is None:
                        producer = model.find_producer(n.input[2])
                        bias = model.get_initializer(producer.input[0])
                        if bias is None:
                            warnings.warn(f"Could not extract bias from node")
                            continue

                    if producer is not None:
                        # Mark the producer node for removal
                        nodes_to_remove.append(producer)
                    
                    # Insert bias as Add node behind the Conv node
                    out_shape = model.get_tensor_shape(n.output[0])
                    # Reshape bias tensor
                    add_shape = [1] * len(out_shape)
                    # ToDo: this must change to "add_shape[-1] = bias.shape[0]" when
                    #  the channels last layout comes around.
                    bias_shape = model.get_tensor_shape(n.input[2])
                    add_shape[1] = bias_shape[0]
                    if bias is not None:
                        model.set_initializer(n.input[2], bias.reshape(add_shape))

                    act_add_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        model.get_tensor_valueinfo(n.output[0]).type.tensor_type.elem_type,
                        out_shape,
                    )
                    graph.value_info.append(act_add_tensor)

                    add_node = helper.make_node(
                        "Add",
                        [act_add_tensor.name, n.input[2]],
                        [n.output[0]],
                    )
                    graph.node.insert(node_ind, add_node)

                    # Repoint Conv output and remove bias tensor
                    n.output[0] = act_add_tensor.name
                    n.input.remove(n.input[2])
                    
                    for node_to_remove in nodes_to_remove:
                        graph.node.remove(node_to_remove)

                    return model, True

        return model, False
