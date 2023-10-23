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

# Protobuf onnx graph node type
from onnx import NodeProto

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# QONNX graph transformations base class
from qonnx.transformation.base import Transformation

#  Gets items from protobuf by name
from qonnx.util.basic import get_by_name


# Tests whether a node is a quant-init, i.e., a quantizer with only initializer
# inputs
def is_quant_init(node: NodeProto, model: ModelWrapper):
    # Only handle existing Quant or BipolarQuant type nodes
    if node is not None and node.op_type in {"Quant", "BipolarQuant"}:
        # All inputs must have initializers, otherwise this is just a normal
        # quant, but not a quant-init
        return all(model.get_initializer(i) is not None for i in node.input)
    # Did not match the operator type
    return False


# Transpose nodes can be folded into quantized initializers, i.e., Quant nodes
# where *all* inputs are initializers. Initializers are constants and part of
# the model graph and thus can be transposed offline.
class FoldTransposeIntoQuantInit(Transformation):
    """
    Fuses a Transpose node into the initializers of a Quant node.
    """

    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # This transformation is triggered by finding a Transpose node
            if node.op_type == "Transpose":
                # Get the predecessors feeding into the transpose node
                predecessors = model.find_direct_predecessors(node)
                # The transform applies only to transpose with exactly one input
                if predecessors is None or len(predecessors) != 1:
                    # Note: Softly skip this node, maybe consider a hard failure
                    #   at least in case there are multiple inputs?
                    continue
                # Check whether the predecessor is a quantizer with only
                # initializer inputs
                if is_quant_init(predecessors[0], model):
                    # Alias to the single predecessor node
                    quant_init = predecessors[0]
                    # Get the (optional) permutation indices of the transpose in
                    # case it is a multi-axis transpose
                    perm = get_by_name(node.attribute, "perm")
                    # Convert permutation indices to list of integers if it is
                    # given
                    perm = perm.ints if perm is not None else None
                    # Transpose all(!) initializer inputs of the quant node
                    for i in quant_init.input:
                        # Get the initializer tensor
                        # Note: No need to validate the presence of the
                        # initializer here, as we already tested this as the
                        # applicability condition above
                        tensor = model.get_initializer(i)
                        # Skip transposing the initializer if the number of
                        # dimensions do not match
                        if perm is not None and len(perm) != tensor.ndim:
                            # Note: Soft skip ok or is this an error?
                            continue
                        # Transpose the tensor, optionally according to the
                        # permutation indices (perm might be None)
                        tensor = tensor.transpose(perm)
                        # Reassign the transposed initializer tensor
                        model.set_initializer(i, tensor)
                        # The graph has been modified, this needs to be reported
                        # back to the caller
                        graph_modified = True
                    # Rewire the graph to skip the transpose node
                    quant_init.output[0] = node.output[0]
                    # Remove the now absorbed transpose node
                    graph.node.remove(node)
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified
