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
import onnx.helper as oh
import onnx.numpy_helper as np_helper
from onnx import TensorProto
from pkgutil import get_data

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model


def test_conv_lowering_convmnist():
    # load the onnx model
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    # model = model.transform(InferShapes())
    # model = model.transform(FoldConstants())
    raw_i = get_data("qonnx.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    input_tensor = np_helper.to_array(input_tensor)
    # execute imported model to get expected answer
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    input_dict = {input_name: input_tensor}
    output_dict_e = oxe.execute_onnx(model, input_dict)
    expected = output_dict_e[output_name]
    # execute transformed model and compare
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(InferShapes())
    output_dict_p = oxe.execute_onnx(model, input_dict)
    produced = output_dict_p[output_name]
    assert np.isclose(produced, expected).all()


# input datatype
@pytest.mark.parametrize("idt", [DataType["INT2"], DataType["INT4"]])
# kernel size
@pytest.mark.parametrize("k_h", [2, 3])
@pytest.mark.parametrize("k_w", [2, 3, 1])
# input dimension
@pytest.mark.parametrize("ifm_dim_h", [9, 11])
@pytest.mark.parametrize("ifm_dim_w", [9, 11, 1])
# input channels
@pytest.mark.parametrize("ifm_ch", [2, 3])
# stride
@pytest.mark.parametrize("stride", [[1, 1], [1, 2], [2, 1], [2, 2]])
# padding
@pytest.mark.parametrize("padding", [[0, 0, 0, 0], [1, 1, 1, 1]])
# dilations
@pytest.mark.parametrize("dilations", [[1, 1], [2, 2], [3, 3]])
# depthwise or channelwise
@pytest.mark.parametrize("dw", [True, False])
# conv bias
@pytest.mark.parametrize("bias", [True, False])
def test_dws_reg_conv_lowering(idt, k_h, k_w, ifm_dim_h, ifm_dim_w, ifm_ch, stride, padding, dilations, dw, bias):
    if k_h > ifm_dim_h:
        pytest.skip("Kernel height must be smaller than image height")
    if k_w > ifm_dim_w:
        pytest.skip("Kernel width must be smaller than image height")
    # Ensure the right padding parameters are set
    if ifm_dim_w == 1:
        dilations[1] = 1
        padding[1] = 0
        padding[3] = 0

    wdt = idt
    odt = DataType["INT32"]
    ofm_ch = ifm_ch
    pad_h = padding[0] + padding[2]
    pad_w = padding[1] + padding[3]
    stride_h = stride[0]
    stride_w = stride[1]

    ofm_dim_h = compute_conv_output_dim(
        ifm_dim_h,
        k_h,
        stride_h,
        pad_h,
        dilations[0],
    )
    ofm_dim_w = compute_conv_output_dim(
        ifm_dim_w,
        k_w,
        stride_w,
        pad_w,
        dilations[1],
    )

    # set up onnx model
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim_h, ifm_dim_w])
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ofm_ch, ofm_dim_h, ofm_dim_w])

    if dw is True:
        W = oh.make_tensor_value_info("W", TensorProto.FLOAT, [ofm_ch, 1, k_h, k_w])
        group = ifm_ch
    else:
        W = oh.make_tensor_value_info("W", TensorProto.FLOAT, [ofm_ch, ifm_ch, k_h, k_w])
        group = 1

    if bias is True:
        B = oh.make_tensor_value_info("B", TensorProto.FLOAT, [ofm_ch])

    dw_cnv = oh.make_node(
        "Conv",
        inputs=["inp", "W"] if not bias else ["inp", "W", "B"],
        outputs=["outp"],
        kernel_shape=[k_h, k_w],
        pads=padding,
        strides=[stride_h, stride_w],
        group=group,
        dilations=dilations,
    )
    graph = oh.make_graph(
        nodes=[dw_cnv],
        name="dw_cnv_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[W] if not bias else [W, B],
    )

    model = qonnx_make_model(graph, producer_name="test_dws_reg_cnv-model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("W", wdt)

    if dw is True:
        w_tensor = gen_finn_dt_tensor(wdt, [ofm_ch, 1, k_h, k_w])
    else:
        w_tensor = gen_finn_dt_tensor(wdt, [ofm_ch, ifm_ch, k_h, k_w])

    if bias is True:
        b_tensor = gen_finn_dt_tensor(odt, [ofm_ch])
        model.set_initializer("B", b_tensor)
        model.set_tensor_datatype("B", odt)

    model.set_initializer("W", w_tensor)
    model = model.transform(InferShapes())

    input_tensor = gen_finn_dt_tensor(idt, [1, ifm_ch, ifm_dim_h, ifm_dim_w])
    input_dict = {"inp": input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    expected = output_dict["outp"]

    model = model.transform(LowerConvsToMatMul())
    assert len(model.get_nodes_by_op_type("Conv")) == 0, "Found Conv nodes after lowering"
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict["outp"]
    assert (produced == expected).all()

    if dw is True:
        # check if created nodes have attributes that indicate depthwise conv
        assert model.get_tensor_sparsity("W") is not None
        im2col_node = getCustomOp(model.graph.node[1])
        assert im2col_node.get_nodeattr("depthwise") == 1


# input datatype
@pytest.mark.parametrize("idt", [DataType["INT2"], DataType["INT4"]])
# kernel size
@pytest.mark.parametrize("k_h", [2])
@pytest.mark.parametrize("k_w", [2])
# input dimension
@pytest.mark.parametrize("ifm_dim_h", [4])
@pytest.mark.parametrize("ifm_dim_w", [4])
# input channels
@pytest.mark.parametrize("ifm_ch", [2])
# stride
@pytest.mark.parametrize("stride", [1, 2])
# padding. Padding is applied to dimensions H and W as: [H_begin, W_begin, H_end, W_end]
@pytest.mark.parametrize(
    "padding",
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ],
)
def test_non_equal_padding(idt, k_h, k_w, ifm_dim_h, ifm_dim_w, ifm_ch, stride, padding):
    wdt = idt
    odt = DataType["INT32"]
    ofm_ch = ifm_ch
    pad_h = padding[0] + padding[2]
    pad_w = padding[1] + padding[3]

    ofm_dim_h = compute_conv_output_dim(
        ifm_dim_h,
        k_h,
        stride,
        pad_h,
    )
    ofm_dim_w = compute_conv_output_dim(
        ifm_dim_w,
        k_w,
        stride,
        pad_w,
    )

    # set up onnx model
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim_h, ifm_dim_w])
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ofm_ch, ofm_dim_h, ofm_dim_w])

    W = oh.make_tensor_value_info("W", TensorProto.FLOAT, [ofm_ch, ifm_ch, k_h, k_w])

    dw_cnv = oh.make_node(
        "Conv",
        inputs=["inp", "W"],
        outputs=["outp"],
        kernel_shape=[k_h, k_w],
        pads=padding,
        strides=[stride, stride],
        group=1,
    )
    graph = oh.make_graph(
        nodes=[dw_cnv],
        name="dw_cnv_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[W],
    )

    model = qonnx_make_model(graph, producer_name="dws_cnv-model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("W", wdt)
    w_tensor = gen_finn_dt_tensor(wdt, [ofm_ch, ifm_ch, k_h, k_w])
    model.set_initializer("W", w_tensor)
    model = model.transform(InferShapes())

    input_tensor = gen_finn_dt_tensor(idt, [1, ifm_ch, ifm_dim_h, ifm_dim_w])
    input_dict = {"inp": input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    expected = output_dict["outp"]

    model = model.transform(LowerConvsToMatMul())
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict["outp"]
    assert (produced == expected).all()


def test_conv_lowering_conv_1x1():
    np.random.seed(0)

    in_feature_dim_h = 7
    in_feature_dim_w = 7
    in_chn = 3
    kernel_size = 1
    out_feature_dim_h = in_feature_dim_h
    out_feature_dim_w = in_feature_dim_w

    input_shape = [1, in_chn, in_feature_dim_h, in_feature_dim_w]
    output_shape = [1, in_chn, out_feature_dim_h, out_feature_dim_w]

    conv_param_shape = [in_chn, in_chn, kernel_size, kernel_size]

    conv_config = {}
    conv_config["dilations"] = [1, 1]
    conv_config["group"] = 1
    conv_config["kernel_shape"] = [kernel_size, kernel_size]
    conv_config["pads"] = [0, 0, 0, 0]
    conv_config["strides"] = [1, 1]

    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)

    value_info = [oh.make_tensor_value_info("p1", TensorProto.FLOAT, conv_param_shape)]

    modelproto = qonnx_make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[oh.make_node("Conv", ["top_in", "p1"], ["top_out"], **conv_config)],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    model.set_initializer("p1", np.random.rand(*conv_param_shape).astype(np.float32))

    new_model = model.transform(LowerConvsToMatMul())
    inp_dict = {"top_in": np.random.rand(*input_shape).astype(np.float32)}

    assert oxe.compare_execution(model, new_model, inp_dict)
    assert new_model.graph.node[0].op_type == "Transpose"
    assert new_model.graph.node[1].op_type == "MatMul"
    assert new_model.graph.node[2].op_type == "Transpose"
    assert len(new_model.graph.node) == 3
