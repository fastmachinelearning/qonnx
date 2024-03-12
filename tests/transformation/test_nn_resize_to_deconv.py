# Copyright (c) 2024, Advanced Micro Devices, Inc.
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
import onnx.numpy_helper as nph
import onnx.parser as oprs
from onnx.checker import check_model
from pkgutil import get_data

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.quant import quant
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.resize_conv_to_deconv import ResizeConvolutionToDeconvolution
from qonnx.util.basic import gen_finn_dt_tensor

np.random.seed(0)


@pytest.mark.parametrize("maintain_bit_width", [True, False])
def test_resize_conv_to_deconv_float_model(maintain_bit_width: bool):
    raw_m = get_data("qonnx.data", "onnx/bsd300x3-espcn/nn_resize/float_model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(InferShapes())
    iname = model.graph.input[0].name
    oname = model.graph.output[0].name
    ishape = model.get_tensor_shape(iname)
    rand_inp = gen_finn_dt_tensor(DataType["FLOAT32"], ishape)
    input_dict = {iname: rand_inp}
    expected = oxe.execute_onnx(model, input_dict)[oname]
    new_model = model.transform(ResizeConvolutionToDeconvolution(maintain_bit_width=maintain_bit_width))
    # check that there are no Resize ops left
    op_types = list(map(lambda x: x.op_type, new_model.graph.node))
    assert "Resize" not in op_types, "Error: the Resize nodes should be removed."
    produced = oxe.execute_onnx(new_model, input_dict)[oname]
    assert np.isclose(expected, produced, atol=1e-4).all(), "Error: expected output does not match the produced output."


@pytest.mark.parametrize("maintain_bit_width", [True, False])
def test_resize_conv_to_deconv_quant_model(maintain_bit_width: bool):
    # get raw quantized model with reference input
    raw_i = get_data("qonnx.data", "onnx/bsd300x3-espcn/test_data/input_0.pb")
    raw_m = get_data("qonnx.data", "onnx/bsd300x3-espcn/nn_resize/quant_model.onnx")
    # create model from the onnx file and infer the shapes
    model = ModelWrapper(raw_m)
    model = model.transform(InferShapes())
    iname = model.graph.input[0].name
    oname = model.graph.output[0].name
    ishape = model.get_tensor_shape(iname)
    # load the reference input tensor
    input_tensor = onnx.load_tensor_from_string(raw_i)
    input_tensor = nph.to_array(input_tensor)
    assert list(input_tensor.shape) == ishape, "Error: reference input doesn't match loaded model."
    input_dict = {iname: input_tensor}
    # get the output from the sub-pixel convolution model
    output_resize_conv = oxe.execute_onnx(model, input_dict)[oname]
    # translate the sub-pixel convolution to the deconvolution
    new_model = model.transform(ResizeConvolutionToDeconvolution(maintain_bit_width=maintain_bit_width))
    # check that there are no Resize ops left
    op_types = list(map(lambda x: x.op_type, new_model.graph.node))
    assert "Resize" not in op_types, "Error: the Resize nodes should be removed."
    # get the output from the deconvolution model
    output_deconv = oxe.execute_onnx(new_model, input_dict)[oname]
    # maintaining the specified bit width introduces additional clipping errors that
    # shouldn't be expected to maintain reasonable functional similarity
    if not maintain_bit_width:
        assert np.isclose(
            output_deconv, output_resize_conv, atol=1 / 255.0, rtol=1.0
        ).all(), "Error: expected output does not match the produced output."


def float_nn_resize_model(r: int, ifm: int, ich: int, och: int, ksize: int, use_bias: bool):
    assert isinstance(ksize, int), "Assuming square kernels, so kernel_size needs to be an int."
    pad = (ksize - 1) // 2

    ishp = (1, ich, ifm, ifm)
    oshp = (1, och, ifm * r, ifm * r)
    wshp = (och, ich, ksize, ksize)
    bshp = (och,)
    rscales = np.array([1.0, 1.0, r, r], dtype=np.float32)
    weight = np.random.randn(*wshp)
    bias = np.random.randn(*bshp)
    ishp_str = str(list(ishp))
    oshp_str = str(list(oshp))
    wshp_str = str(list(wshp))
    bshp_str = str(list(bshp))

    if use_bias:
        params_str = f"""
        <
            float{wshp_str} conv_param,
            float{bshp_str} bias_param,
            float roi,
            float scales
        >
        """
    else:
        params_str = f"""
        <
            float{wshp_str} conv_param,
            float roi,
            float scales
        >
        """

    if use_bias:
        conv_str = f"""
            out0 = Conv<
                dilations=[1,1],
                group=1,
                kernel_shape=[{ksize},{ksize}],
                strides=[1,1],
                pads=[{pad},{pad},{pad},{pad}]
            >(hid0, conv_param, bias_param)
        """
    else:
        conv_str = f"""
            out0 = Conv<
                dilations=[1,1],
                group=1,
                kernel_shape=[{ksize},{ksize}],
                strides=[1,1],
                pads=[{pad},{pad},{pad},{pad}]
            >(hid0, conv_param)
        """

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 13]
    >
    agraph (float{ishp_str} in0) => (float{oshp_str} out0)
    {params_str}
    {{
        hid0 = Resize<
            mode="nearest"
        >(in0, roi, scales)
        {conv_str}
    }}
    """

    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model.set_initializer("roi", np.empty(0))
    model.set_initializer("scales", rscales.astype(np.float32))
    model.set_initializer("conv_param", weight.astype(np.float32))
    if use_bias:
        model.set_initializer("bias_param", bias.astype(np.float32))
    model = model.transform(InferShapes())
    check_model(model._model_proto)
    return model


def quant_nn_resize_model(r: int, ifm: int, ich: int, och: int, ksize: int, use_bias: bool, channelwise: bool):
    assert isinstance(ksize, int), "Assuming square kernels, so kernel_size needs to be an int."
    pad = (ksize - 1) // 2

    ishp = (1, ich, ifm, ifm)
    oshp = (1, och, ifm * r, ifm * r)
    wshp = (och, ich, ksize, ksize)
    bshp = (och,)
    rscales = np.array([1.0, 1.0, r, r], dtype=np.float32)
    weight = np.random.randn(*wshp)
    bias = np.random.randn(*bshp)
    ishp_str = str(list(ishp))
    oshp_str = str(list(oshp))
    wshp_str = str(list(wshp))
    bshp_str = str(list(bshp))

    if channelwise:
        q_attr_shp = (och, 1, 1, 1)
    else:
        q_attr_shp = (1,)
    attrshp_str = str(list(q_attr_shp))
    scale = np.random.rand(*q_attr_shp).astype(np.float32)
    zeropt = np.zeros(q_attr_shp).astype(np.float32)  # NOTE: needs to be integer
    bitwidth = np.array(4.0)

    weight: np.ndarray = quant(weight, scale, zeropt, bitwidth, signed=True, narrow=True, rounding_mode="ROUND")

    if use_bias:
        params_str = f"""
        <
            float{wshp_str} conv_param,
            float{attrshp_str} scale_param,
            float{attrshp_str} zeropt_param,
            float{bshp_str} bias_param,
            float bitwidth_param,
            float scale_bias,
            float zeropt_bias,
            float bitwidth_bias,
            float roi,
            float scales
        >
        """
    else:
        params_str = f"""
        <
            float{wshp_str} conv_param,
            float{attrshp_str} scale_param,
            float{attrshp_str} zeropt_param,
            float roi,
            float scales,
            float bitwidth_param
        >
        """

    if use_bias:
        scale_bias = np.random.rand(
            1,
        )
        zeropt_bias = np.array(0.0)
        bitwidth_bias = np.array(16.0)
        convs_str = f"""
        param1 = qonnx.custom_op.general.Quant<
            signed=1,
            narrow=1,
            rounding_mode="ROUND"
        >(bias_param, scale_bias, zeropt_bias, bitwidth_bias)
        out0 = Conv<
            dilations=[1,1],
            group=1,
            kernel_shape=[{ksize},{ksize}],
            strides=[1,1],
            pads=[{pad},{pad},{pad},{pad}]
        >(hid0, param0, param1)
        """
    else:
        convs_str = f"""
        out0 = Conv<
            dilations=[1,1],
            group=1,
            kernel_shape=[{ksize},{ksize}],
            strides=[1,1],
            pads=[{pad},{pad},{pad},{pad}]
        >(hid0, param0)
        """

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 13, "qonnx.custom_op.general" : 1]
    >
    agraph (float{ishp_str} in0) => (float{oshp_str} out0)
    {params_str}
    {{
        hid0 = Resize<
            mode="nearest"
        >(in0, roi, scales)
        param0 = qonnx.custom_op.general.Quant<
            signed=1,
            narrow=1,
            rounding_mode="ROUND"
        >(conv_param, scale_param, zeropt_param, bitwidth_param)
        {convs_str}
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model.set_initializer("roi", np.empty(0))
    model.set_initializer("scales", rscales.astype(np.float32))
    model.set_initializer("conv_param", weight.astype(np.float32))
    if use_bias:
        model.set_initializer("bias_param", bias.astype(np.float32))
        model.set_initializer("scale_bias", scale_bias.astype(np.float32))
        model.set_initializer("zeropt_bias", zeropt_bias.astype(np.float32))
        model.set_initializer("bitwidth_bias", bitwidth_bias.astype(np.float32))
    model.set_initializer("scale_param", scale.astype(np.float32))
    model.set_initializer("zeropt_param", zeropt.astype(np.float32))
    model.set_initializer("bitwidth_param", bitwidth.astype(np.float32))
    model = model.transform(InferShapes())
    check_model(model._model_proto)
    return model


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
@pytest.mark.parametrize("upscale_factor", [1, 2, 3, 4])
@pytest.mark.parametrize("bias", [True, False])
def test_float_resize_conv_to_deconv_layer(kernel_size: int, upscale_factor: int, bias: bool):
    och = 10  # output channels
    ich = 3  # input channels
    ifm = 4  # input feature map size
    input_shape = [1, ich, ifm, ifm]
    # Create resize convolution layer that upsamples a 4x4 image with 1 I/O channel
    model_1 = float_nn_resize_model(upscale_factor, ifm, ich, och, kernel_size, bias)
    model_2 = model_1.transform(ResizeConvolutionToDeconvolution())
    inp_dict = {"inp": np.random.rand(*input_shape).astype(np.float32)}
    assert oxe.compare_execution(model_1, model_2, inp_dict)


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
@pytest.mark.parametrize("upscale_factor", [1, 2, 3, 4])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("channelwise", [True, False])
@pytest.mark.parametrize("maintain_bit_width", [True, False])
def test_quant_resize_conv_to_deconv_layer(
    kernel_size: int, upscale_factor: int, bias: bool, channelwise: bool, maintain_bit_width: bool
):
    och = 10  # output channels
    ich = 3  # input channels
    ifm = 4  # input feature map size
    input_shape = [1, ich, ifm, ifm]
    # Create resize convolution layer that upsamples a 4x4 image with 1 I/O channel
    model_1 = quant_nn_resize_model(upscale_factor, ifm, ich, och, kernel_size, bias, channelwise)
    model_2 = model_1.transform(ResizeConvolutionToDeconvolution(maintain_bit_width=maintain_bit_width))
    inp_dict = {"inp": np.random.rand(*input_shape).astype(np.float32)}
    assert oxe.compare_execution(model_1, model_2, inp_dict)

    if maintain_bit_width:
        bw1 = model_1.get_initializer("bitwidth_param")
        bw2 = model_2.get_initializer("bitwidth_param")
        assert (bw1 == bw2).all()
