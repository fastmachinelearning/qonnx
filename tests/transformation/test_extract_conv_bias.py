# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
import onnx.helper as oh
from onnx import TensorProto

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model


# depthwise or channelwise
@pytest.mark.parametrize("dw", [True, False])
# conv bias
@pytest.mark.parametrize(
    "bias", ["float", "int_quant_per_tensor", "int_quant_per_channel", "bp_quant_per_tensor", "bp_quant_per_channel", None]
)
def test_extract_conv_bias(dw, bias):
    ishape = (1, 32, 111, 111)
    if dw is True:
        group = ishape[1]
        out_channels = ishape[1]
        kernel_size = 3
        padding = 1
        stride = 1
        w_shape = (32, 1, 3, 3)

    else:
        group = 1
        out_channels = 64
        kernel_size = 1
        padding = 0
        stride = 1
        w_shape = (64, 32, 1, 1)

    wdt = idt = odt = DataType["FLOAT32"]

    # set up onnx model
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, [ishape[0], out_channels, ishape[2], ishape[3]])

    W = oh.make_tensor_value_info("W", TensorProto.FLOAT, w_shape)

    if bias is not None:
        bias_shape = (out_channels,)
        if "quant_per_channel" in bias:
            scale_shape = (out_channels,)
        elif "quant_per_tensor" in bias:
            scale_shape = (1,)
        B = oh.make_tensor_value_info("B", TensorProto.FLOAT, bias_shape)

    cnv_node = oh.make_node(
        "Conv",
        inputs=["inp", "W"] if not bias else ["inp", "W", "B"],
        outputs=["outp"],
        kernel_shape=[kernel_size, kernel_size],
        pads=[padding, padding, padding, padding],
        strides=[stride, stride],
        group=group,
    )
    nodes = [cnv_node]
    value_info = [W] if not bias else [W, B]
    # if the bias isn't quantized, we can directly wire up the Conv layer
    # otherwise an additional Quant node needs to be inserted
    if bias is not None and "quant" in bias:
        if "bp" in bias:
            optype = "BipolarQuant"
        elif "int" in bias:
            optype = "IntQuant"
        # inputs to Quant node
        param0 = oh.make_tensor_value_info("param0", TensorProto.FLOAT, bias_shape)
        param1 = oh.make_tensor_value_info("param1", TensorProto.FLOAT, scale_shape)
        param2 = oh.make_tensor_value_info("param2", TensorProto.FLOAT, [1])
        value_info.append(param0)
        value_info.append(param1)
        value_info.append(param2)
        if "int" in bias:
            param3 = oh.make_tensor_value_info("param3", TensorProto.FLOAT, [1])
            value_info.append(param3)
        quant_node = oh.make_node(
            optype,
            domain="qonnx.custom_op.general",
            inputs=["param0", "param1", "param2", "param3"] if "int" in bias else ["param0", "param1", "param2"],
            outputs=["B"],
            narrow=0,
            rounding_mode="ROUND",
            signed=1,
        )
        nodes.append(quant_node)
    graph = oh.make_graph(
        nodes=nodes,
        name="cnv_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=value_info,
    )

    model = qonnx_make_model(graph, producer_name="test-cnv-model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("W", wdt)

    w_tensor = gen_finn_dt_tensor(wdt, w_shape)

    if bias is not None:
        b_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], bias_shape)
        # set B tensor directly or set first input of quant node
        if "quant" in bias:
            model.set_initializer("param0", b_tensor)
            scale = gen_finn_dt_tensor(DataType["FLOAT32"], bias_shape)
            model.set_initializer("param1", scale)
            model.set_initializer("param2", np.zeros(1))
            if "int" in bias:
                model.set_initializer("param3", 8 * np.ones(1))
        else:
            model.set_initializer("B", b_tensor)

    model.set_initializer("W", w_tensor)
    model = model.transform(InferShapes())

    input_tensor = gen_finn_dt_tensor(idt, ishape)
    output_dict = oxe.execute_onnx(model, {model.graph.input[0].name: input_tensor})
    expected = output_dict[model.graph.output[0].name]

    model = model.transform(ExtractBiasFromConv())

    if bias is not None:
        assert len(model.get_nodes_by_op_type("Add")) > 0, "Bias wasn't extracted into add node"

    output_dict = oxe.execute_onnx(model, {model.graph.input[0].name: input_tensor})
    produced = output_dict[model.graph.output[0].name]

    # check if is close (fp calculation)
    assert np.isclose(produced, expected, atol=1e-3).all()


# conv transpose bias
@pytest.mark.parametrize(
    "bias", ["float", "int_quant_per_tensor", "int_quant_per_channel", "bp_quant_per_tensor", "bp_quant_per_channel", None]
)
def test_extract_conv_transpose_bias(bias):
    ishape = (1, 32, 111, 111)
    group = 1
    out_channels = 64
    kernel_size = 1
    padding = 0
    stride = 1
    w_shape = (32, 64, 1, 1)

    wdt = idt = odt = DataType["FLOAT32"]

    # Set up ONNX model
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp_shape = (ishape[0], out_channels, ishape[2], ishape[3])
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, outp_shape)

    W = oh.make_tensor_value_info("W", TensorProto.FLOAT, w_shape)

    if bias is not None:
        bias_shape = (out_channels,)
        if "quant_per_channel" in bias:
            scale_shape = (out_channels,)
        elif "quant_per_tensor" in bias:
            scale_shape = (1,)
        B = oh.make_tensor_value_info("B", TensorProto.FLOAT, bias_shape)

    cnv_node = oh.make_node(
        "ConvTranspose",
        inputs=["inp", "W"] if not bias else ["inp", "W", "B"],
        outputs=["outp"],
        kernel_shape=[kernel_size, kernel_size],
        pads=[padding, padding, padding, padding],
        strides=[stride, stride],
        group=group,
    )
    nodes = [cnv_node]
    value_info = [W] if not bias else [W, B]

    # If the bias isn't quantized, we can directly wire up the ConvTranspose layer
    # Otherwise, an additional Quant node needs to be inserted
    if bias is not None and "quant" in bias:
        if "bp" in bias:
            optype = "BipolarQuant"
        elif "int" in bias:
            optype = "IntQuant"
        # Inputs to Quant node
        param0 = oh.make_tensor_value_info("param0", TensorProto.FLOAT, bias_shape)
        param1 = oh.make_tensor_value_info("param1", TensorProto.FLOAT, scale_shape)
        param2 = oh.make_tensor_value_info("param2", TensorProto.FLOAT, [1])
        value_info.append(param0)
        value_info.append(param1)
        value_info.append(param2)
        if "int" in bias:
            param3 = oh.make_tensor_value_info("param3", TensorProto.FLOAT, [1])
            value_info.append(param3)
        quant_node = oh.make_node(
            optype,
            domain="qonnx.custom_op.general",
            inputs=["param0", "param1", "param2", "param3"] if "int" in bias else ["param0", "param1", "param2"],
            outputs=["B"],
            narrow=0,
            rounding_mode="ROUND",
            signed=1,
        )
        nodes.append(quant_node)

    graph = oh.make_graph(
        nodes=nodes,
        name="cnv_transpose_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=value_info,
    )

    model = qonnx_make_model(graph, producer_name="test-cnv-transpose-model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("W", wdt)

    w_tensor = gen_finn_dt_tensor(wdt, w_shape)

    if bias is not None:
        b_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], bias_shape)
        # Set B tensor directly or set first input of quant node
        if "quant" in bias:
            model.set_initializer("param0", b_tensor)
            scale = gen_finn_dt_tensor(DataType["FLOAT32"], bias_shape)
            model.set_initializer("param1", scale)
            model.set_initializer("param2", np.zeros(1))
            if "int" in bias:
                model.set_initializer("param3", 8 * np.ones(1))
        else:
            model.set_initializer("B", b_tensor)

    model.set_initializer("W", w_tensor)
    model = model.transform(InferShapes())

    input_tensor = gen_finn_dt_tensor(idt, ishape)
    output_dict = oxe.execute_onnx(model, {model.graph.input[0].name: input_tensor})
    expected = output_dict[model.graph.output[0].name]

    model = model.transform(ExtractBiasFromConv())

    if bias is not None:
        assert len(model.get_nodes_by_op_type("Add")) > 0, "Bias wasn't extracted into add node"

    output_dict = oxe.execute_onnx(model, {model.graph.input[0].name: input_tensor})
    produced = output_dict[model.graph.output[0].name]

    # Check if is close (fp calculation)
    assert np.isclose(produced, expected, atol=1e-3).all()
