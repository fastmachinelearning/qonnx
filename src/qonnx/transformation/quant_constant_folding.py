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

import warnings

from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


class FoldTransposeIntoQuantInit(Transformation):
    """
    Fueses a Transpose node into the initalizer of a Quant node.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        # Find transpose nodes, which have Quant node with initilizer upstream.
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Transpose":
                predecessors = model.find_direct_predecessors(n)
                # Check if we reached the top of the graph
                if predecessors is None:
                    continue
                predecessor = predecessors[0]
                if predecessor.op_type == "Quant" or predecessor.op_type == "BipolarQuant":
                    for inp in predecessor.input:
                        if not isinstance(model.get_initializer(inp), type(None)):
                            # Explicitly apply the transpose to the initializers
                            # of the previous node
                            target_tensor = model.get_initializer(inp)
                            if target_tensor is None:
                                warnings.warn(
                                    f"Cannot fold transpose {n} into Quant/BipolarQuant node {predecessor}, "
                                    f"due to not initialized tensor: {inp}. "
                                    f"Exiting FoldTransposeIntoQuantInit transformation."
                                )
                                return model, False
                            # Make sure the tensor has the correct shape
                            perm = get_by_name(n.attribute, "perm")
                            if perm is None:
                                target_tensor = target_tensor.transpose()
                                model.set_initializer(inp, target_tensor)
                                graph_modified = True
                            elif len(perm.ints) == len(target_tensor.shape):
                                target_tensor = target_tensor.transpose(perm.ints)
                                model.set_initializer(inp, target_tensor)
                                graph_modified = True
                    # Reconnect predecessor and delete transpose node
                    predecessor.output[0] = n.output[0]
                    graph.node.remove(n)

                    return model, graph_modified

        return model, graph_modified
