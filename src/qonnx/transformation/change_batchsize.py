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

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation


class ChangeBatchSize(Transformation):
    """Change the batch size dimension to the given value for the entire graph
    by changing it for the global input/output and removing all intermediate
    shapes (will need a call to shape inference to restore shapes).
    Will attempt to handle any Reshape nodes with constant shape parameters by
    changing the batch size dimension value in the parameter."""

    def __init__(self, bsize):
        super().__init__()
        self.bsize = int(bsize)

    def apply(self, model: ModelWrapper):
        onnx_model = model.model
        bsize = self.bsize
        onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = bsize
        onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = bsize
        while len(onnx_model.graph.value_info) > 0:
            onnx_model.graph.value_info.remove(onnx_model.graph.value_info[0])
        reshape_nodes = model.get_nodes_by_op_type("Reshape")
        for reshape_node in reshape_nodes:
            rs_param_name = reshape_node.input[1]
            rs_param = model.get_initializer(rs_param_name)
            if rs_param is not None:
                rs_param = rs_param.copy()
                rs_param[0] = bsize
                model.set_initializer(rs_param_name, rs_param)
        return (model, False)
