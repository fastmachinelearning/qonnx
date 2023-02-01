# Copyright (c) 2020 Xilinx, Inc.
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
import pytest

from onnx import TensorProto, helper

import qonnx.core.data_layout as DataLayout
import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.maxpoolnhwc import compute_pool_output_dim
from qonnx.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, get_by_name, qonnx_make_model


# stride
@pytest.mark.parametrize("s", [1, 2])
# kernel
@pytest.mark.parametrize("k", [3, 4])
# ibits
@pytest.mark.parametrize("ibits", [4, 8])
# obits
@pytest.mark.parametrize("obits", [2, 4])
# signed
@pytest.mark.parametrize("signed", [False, True])
# channels
@pytest.mark.parametrize("c", [2, 3])
# input dimension
@pytest.mark.parametrize("idim", [6, 7])
def test_change_datalayout_quantavgpool(s, k, ibits, obits, signed, c, idim):
    n = 1
    odim = compute_pool_output_dim(idim, k, s)
    # determine input QONNX datatype
    if signed is True:
        prefix = "INT"
    else:
        prefix = "UINT"
    dt_name = prefix + str(ibits)
    dtype = DataType[dt_name]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [n, c, idim, idim])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [n, c, odim, odim])

    node = helper.make_node(
        "QuantAvgPool2d",
        ["inp"],
        ["outp"],
        domain="qonnx.custom_op.general",
        stride=s,
        kernel=k,
        ibits=ibits,
        obits=obits,
        signed=signed,
        data_layout="NCHW",
    )
    graph = helper.make_graph(nodes=[node], name="single-quantavgpool", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph)
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model_transformed = model.transform(ChangeDataLayoutQuantAvgPool2d())
    model_transformed = model_transformed.transform(InferShapes())
    model_transformed = model_transformed.transform(InferDataTypes())
    model_transformed = model_transformed.transform(InferDataLayouts())
    model_transformed = model_transformed.transform(GiveUniqueNodeNames())
    model_transformed = model_transformed.transform(GiveReadableTensorNames())
    inp_values = gen_finn_dt_tensor(dtype, [n, c, idim, idim])
    idict = {"inp": inp_values}
    assert oxe.compare_execution(model, model_transformed, idict)
    assert len(model.graph.node) + 2 == len(model_transformed.graph.node)
    assert model_transformed.graph.node[-1].op_type == "Transpose"
    assert model_transformed.graph.node[0].op_type == "Transpose"
    # check if QuantAvgPool2d node has datalayout set correctly
    node = model_transformed.graph.node[1]
    d_layout = get_by_name(node.attribute, "data_layout").s.decode("UTF-8")
    assert d_layout == "NHWC"
    assert model_transformed.get_tensor_layout(node.input[0]) == DataLayout.NHWC
    assert model_transformed.get_tensor_layout(node.output[0]) == DataLayout.NHWC
