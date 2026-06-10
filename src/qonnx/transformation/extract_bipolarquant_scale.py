# Copyright (c) 2026 EmLogic AS
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
from onnx import TensorProto, helper

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueParameterTensors, SortGraph
from qonnx.transformation.remove import RemoveIdentityOps


class ExtractBipolarQuantScale(Transformation):
    """Extract any non-identity scale BipolarQuant input as
    separate Div/Mul nodes preceding and following the BipolarQuant node."""

    def apply(self, model: ModelWrapper):
        graph = model.graph
        for node in graph.node:
            if node.op_type == "BipolarQuant":
                bq_node = node
                input_nm, scale_nm = node.input
                scale_t = model.get_initializer(scale_nm)
                ishp = model.get_tensor_shape(input_nm)
                extract_scale = False
                if scale_t is not None and (scale_t != 1).any():
                    extract_scale = True
                if not extract_scale:
                    continue
                running_input = input_nm
                if extract_scale:
                    # create new Div node that divides the input
                    # by the scale
                    inp_scaled_nm = model.make_new_valueinfo_name()
                    inp_scaled = helper.make_tensor_value_info(
                        inp_scaled_nm,
                        TensorProto.FLOAT,
                        ishp,
                    )
                    graph.value_info.append(inp_scaled)
                    inp_scale_node = helper.make_node("Div", [running_input, scale_nm], [inp_scaled_nm])
                    graph.node.append(inp_scale_node)
                    # remove scale from BipolarQuant node
                    new_scale_nm = model.make_new_valueinfo_name()
                    model.set_initializer(new_scale_nm, np.asarray(1.0, dtype=np.float32))
                    bq_node.input[1] = new_scale_nm
                    running_input = inp_scaled_nm
                # rewire node input to newly created Div node
                bq_node.input[0] = running_input
                final_output = bq_node.output[0]
                if extract_scale:
                    # create new Mul node that applies the output scale
                    out_scale_nm = model.make_new_valueinfo_name()
                    out_scale = helper.make_tensor_value_info(
                        out_scale_nm,
                        TensorProto.FLOAT,
                        ishp,
                    )
                    bq_node.output[0] = out_scale_nm
                    graph.value_info.append(out_scale)
                    out_scale_node = helper.make_node("Mul", [out_scale_nm, scale_nm], [final_output])
                    graph.node.append(out_scale_node)

                if extract_scale:
                    # since we used append() for new nodes, need to call
                    # SortGraph to ensure correct (topological) order
                    model = model.transform(SortGraph())
                    # Remove potential unity multiplications from alpha and beta attributes
                    model = model.transform(RemoveIdentityOps())
                    # Ensure unique parameter tensors
                    model = model.transform(GiveUniqueParameterTensors())
                    return model, True

        return model, False
