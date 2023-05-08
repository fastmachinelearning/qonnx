# Copyright (c) 2020, Xilinx
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

import pytest

import numpy as np
import onnx
import os
import urllib.request as ureq

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

download_url = "https://github.com/onnx/models/raw/main/vision/classification"
download_url += "/shufflenet/model/shufflenet-9.onnx"
export_onnx_path = download_url.split("/")[-1]


def test_batchnorm_to_affine_shufflenet():
    ureq.urlretrieve(download_url, export_onnx_path)
    if not os.path.isfile(export_onnx_path):
        pytest.skip("Couldn't download ONNX model, skipping")
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    iname = model.graph.input[0].name
    oname = model.graph.output[0].name
    ishape = model.get_tensor_shape(iname)
    np.random.seed(0)
    rand_inp = gen_finn_dt_tensor(DataType["INT8"], ishape)
    input_dict = {iname: rand_inp}
    expected = oxe.execute_onnx(model, input_dict)[oname]
    new_model = model.transform(BatchNormToAffine())
    # check that there are no BN nodes left
    op_types = list(map(lambda x: x.op_type, new_model.graph.node))
    assert "BatchNormalization" not in op_types
    produced = oxe.execute_onnx(new_model, input_dict)[oname]
    assert np.isclose(expected, produced, atol=1e-05).all()
    os.remove(export_onnx_path)


@pytest.mark.parametrize("epsilon", [0.0, 0.00001, 0.001])
def test_batchnorm_to_affine_epsilon(epsilon):
    """Dummy batchnorm node to test out the epsilon attribute."""

    batchnorm_node = onnx.helper.make_node(
        "BatchNormalization",
        inputs=["x", "s", "bias", "mean", "var"],
        outputs=["y"],
        epsilon=epsilon,
    )

    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 5, 5])
    s = onnx.helper.make_tensor_value_info("s", onnx.TensorProto.FLOAT, [3])
    bias = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [3])
    mean = onnx.helper.make_tensor_value_info("mean", onnx.TensorProto.FLOAT, [3])
    var = onnx.helper.make_tensor_value_info("var", onnx.TensorProto.FLOAT, [3])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3, 5, 5])

    # Graph
    graph = onnx.helper.make_graph(
        nodes=[batchnorm_node],
        name="test_batchnorm_graph",
        inputs=[x],
        outputs=[y],
        value_info=[s, bias, mean, var],
    )

    onnx_model = qonnx_make_model(graph, producer_name="test_batchnorm-model")
    model = ModelWrapper(onnx_model)

    model.set_initializer("s", np.array([1, 2, 3]).astype(np.float32))
    model.set_initializer("bias", np.array([1, 2, 3]).astype(np.float32))
    model.set_initializer("mean", np.array([3, 4, 5]).astype(np.float32))
    model.set_initializer("var", np.array([0.5, 0.7, 0.3]).astype(np.float32))

    i_val = np.arange(0, 3 * 5 * 5, dtype=np.float32)
    i_val = np.reshape(i_val, [1, 3, 5, 5])
    input_dict = {"x": i_val}
    output_node_name = "y"

    output_dict = oxe.execute_onnx(model, input_dict, return_full_exec_context=True)
    output_original = output_dict[output_node_name]

    model_lowered = model.transform(BatchNormToAffine())
    output_dict = oxe.execute_onnx(model_lowered, input_dict, return_full_exec_context=True)
    output_lowered = output_dict[output_node_name]

    assert (output_original == output_lowered).all()
