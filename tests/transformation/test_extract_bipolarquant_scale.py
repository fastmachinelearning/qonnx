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

import pytest

import numpy as np
import onnx.parser as oprs

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.extract_bipolarquant_scale import ExtractBipolarQuantScale


def make_test_model(ishp, channelwise, need_extraction_scale):
    ishp_str = str(list(ishp))
    if channelwise:
        q_attr_shp = ishp
    else:
        q_attr_shp = (1,)
    attrshp_str = str(list(q_attr_shp))
    np.random.seed(0)
    if need_extraction_scale:
        scale = np.random.rand(*q_attr_shp).astype(np.float32)
    else:
        scale = np.ones(q_attr_shp, dtype=np.float32)

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{ishp_str} in0) => (float{ishp_str} out0)
    <
        float{attrshp_str} scale_param
    >
    {{
        out0 = qonnx.custom_op.general.BipolarQuant(in0, scale_param)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model.set_initializer("scale_param", scale)
    return model


@pytest.mark.parametrize("need_extraction_scale", [True, False])
@pytest.mark.parametrize("channelwise", [True, False])
def test_extract_bipolar_quant_scale(channelwise, need_extraction_scale):
    ishp = (1, 10)
    model = make_test_model(ishp, channelwise, need_extraction_scale)
    ishp = model.get_tensor_shape("in0")
    inp = np.random.rand(*ishp).astype(np.float32)
    y_golden = execute_onnx(model, {"in0": inp})["out0"]
    model_new = model.transform(ExtractBipolarQuantScale())
    y_ret = execute_onnx(model_new, {"in0": inp})["out0"]
    assert np.allclose(y_golden, y_ret)
    qnt_node = model_new.get_nodes_by_op_type("BipolarQuant")[0]
    new_scale = model_new.get_initializer(qnt_node.input[1])
    assert (new_scale == 1).all()
    if need_extraction_scale:
        assert len(model_new.get_nodes_by_op_type("Mul")) == 1
        assert len(model_new.get_nodes_by_op_type("Div")) == 1
    else:
        assert len(model_new.get_nodes_by_op_type("Mul")) == 0
        assert len(model_new.get_nodes_by_op_type("Div")) == 0
