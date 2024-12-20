# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from brevitas.export.onnx import onnx_export_opset
from torch.autograd import Function

AXIS_OPSET = 13
DOMAIN_STRING = "onnx.brevitas"


class DequantizeLinearFn(Function):
    @staticmethod
    def symbolic(g, x, input_scale, input_zero_point, input_axis):
        opset_version = onnx_export_opset()

        if input_axis is not None and opset_version < AXIS_OPSET:
            raise RuntimeError("ONNX Opset 13 is required for per-channel quantization")
        elif input_axis is not None and opset_version >= AXIS_OPSET:
            ret = g.op("DequantizeLinear", x, input_scale, input_zero_point, axis_i=input_axis)
        else:
            ret = g.op("DequantizeLinear", x, input_scale, input_zero_point)
        return ret

    @staticmethod
    def forward(ctx, int_x, input_scale, input_zero_point, input_axis):
        return int_x.float()


class IntClipFn(Function):
    @staticmethod
    def symbolic(g, int_x, min_int_val, max_int_val):
        ret = g.op("Clip", int_x, min_int_val, max_int_val)
        return ret

    @staticmethod
    def forward(ctx, int_x, min_int_val, max_int_val):
        return int_x


class QuantizeLinearFn(Function):
    @staticmethod
    def symbolic(g, x, output_scale, ouput_zero_point, output_dtype, output_axis):
        opset_version = onnx_export_opset()

        if output_axis is not None and opset_version < AXIS_OPSET:
            raise RuntimeError("ONNX Opset 13 is required for per-channel quantization")
        elif output_axis is not None and opset_version >= AXIS_OPSET:
            ret = g.op("QuantizeLinear", x, output_scale, ouput_zero_point, axis_i=output_axis)
        else:
            ret = g.op("QuantizeLinear", x, output_scale, ouput_zero_point)
        return ret

    @staticmethod
    def forward(ctx, x, output_scale, ouput_zero_point, output_dtype, output_axis):
        return x.type(output_dtype)


class BrevitasQuantLSTMCellFn(Function):
    @staticmethod
    def symbolic(
        g,  # args and kwargs passed from _QuantLSTMLayer
        quant_input,
        quant_hidden_state,
        quant_cell_state,
        quant_weight_ii,
        quant_weight_if,
        quant_weight_ic,
        quant_weight_io,
        quant_weight_hi,
        quant_weight_hf,
        quant_weight_hc,
        quant_weight_ho,
        quant_bias_input,
        quant_bias_forget,
        quant_bias_cell,
        quant_bias_output,  # Symbolic kwargs passed from BrevitasQuantLSTMLayerHandler
        batch_first,
        reverse_input,
        cifg,  # Output quant
        output_scale,
        output_zero_point,
        output_bit_width,
        output_narrow_range,
        output_signed,
        output_rounding_mode,  # Cell state quant
        cell_state_scale,
        cell_state_zero_point,
        cell_state_bit_width,
        cell_state_narrow_range,
        cell_state_signed,
        cell_state_rounding_mode,  # Input gate accumulator quant
        input_acc_scale,
        input_acc_zero_point,
        input_acc_bit_width,
        input_acc_narrow_range,
        input_acc_signed,
        input_acc_rounding_mode,  # Forget gate accumulator quant
        forget_acc_scale,
        forget_acc_zero_point,
        forget_acc_bit_width,
        forget_acc_narrow_range,
        forget_acc_signed,
        forget_acc_rounding_mode,  # Cell gate accumulator quant
        cell_acc_scale,
        cell_acc_zero_point,
        cell_acc_bit_width,
        cell_acc_narrow_range,
        cell_acc_signed,
        cell_acc_rounding_mode,  # Output gate accumulator quant
        output_acc_scale,
        output_acc_zero_point,
        output_acc_bit_width,
        output_acc_narrow_range,
        output_acc_signed,
        output_acc_rounding_mode,  # Input gate sigmoid quant
        input_sigmoid_scale,
        input_sigmoid_zero_point,
        input_sigmoid_bit_width,
        input_sigmoid_narrow_range,
        input_sigmoid_signed,
        input_sigmoid_rounding_mode,  # Forget gate sigmoid quant
        forget_sigmoid_scale,
        forget_sigmoid_zero_point,
        forget_sigmoid_bit_width,
        forget_sigmoid_narrow_range,
        forget_sigmoid_signed,
        forget_sigmoid_rounding_mode,  # Cell gate tanh quant
        cell_tanh_scale,
        cell_tanh_zero_point,
        cell_tanh_bit_width,
        cell_tanh_narrow_range,
        cell_tanh_signed,
        cell_tanh_rounding_mode,  # Output gate sigmoid quant
        output_sigmoid_scale,
        output_sigmoid_zero_point,
        output_sigmoid_bit_width,
        output_sigmoid_narrow_range,
        output_sigmoid_signed,
        output_sigmoid_rounding_mode,  # Hidden state tanh quant
        hidden_state_tanh_scale,
        hidden_state_tanh_zero_point,
        hidden_state_tanh_bit_width,
        hidden_state_tanh_narrow_range,
        hidden_state_tanh_signed,
        hidden_state_tanh_rounding_mode,
    ):
        return g.op(
            f"{DOMAIN_STRING}::QuantLSTMCell",  # Tensors
            # Input values
            quant_input,
            quant_hidden_state,
            quant_cell_state,
            quant_weight_ii,
            quant_weight_if,
            quant_weight_ic,
            quant_weight_io,
            quant_weight_hi,
            quant_weight_hf,
            quant_weight_hc,
            quant_weight_ho,
            quant_bias_input,
            quant_bias_forget,
            quant_bias_cell,
            quant_bias_output,  # Output quant
            output_scale,
            output_zero_point,
            output_bit_width,  # Cell state quant
            cell_state_scale,
            cell_state_zero_point,
            cell_state_bit_width,  # Input gate accumulator quant
            input_acc_scale,
            input_acc_zero_point,
            input_acc_bit_width,  # Forget gate accumulator quant
            forget_acc_scale,
            forget_acc_zero_point,
            forget_acc_bit_width,  # Cell gate accumulator quant
            cell_acc_scale,
            cell_acc_zero_point,
            cell_acc_bit_width,  # Output gate accumulator quant
            output_acc_scale,
            output_acc_zero_point,
            output_acc_bit_width,  # Input gate sigmoid quant
            input_sigmoid_scale,
            input_sigmoid_zero_point,
            input_sigmoid_bit_width,  # Forget gate sigmoid quant
            forget_sigmoid_scale,
            forget_sigmoid_zero_point,
            forget_sigmoid_bit_width,  # Cell gate tanh quant
            cell_tanh_scale,
            cell_tanh_zero_point,
            cell_tanh_bit_width,  # Output gate sigmoid quant
            output_sigmoid_scale,
            output_sigmoid_zero_point,
            output_sigmoid_bit_width,  # Hidden state tanh quant
            hidden_state_tanh_scale,
            hidden_state_tanh_zero_point,
            hidden_state_tanh_bit_width,
            # Attributes
            batch_first_i=batch_first,
            reverse_input_i=reverse_input,
            cifg_i=cifg,
            output_narrow_i=output_narrow_range,
            output_signed_i=output_signed,
            output_rounding_mode_s=output_rounding_mode,
            cell_state_narrow_i=cell_state_narrow_range,
            cell_state_signed_i=cell_state_signed,
            cell_state_rounding_mode_s=cell_state_rounding_mode,
            input_acc_narrow_i=input_acc_narrow_range,
            input_acc_signed_i=input_acc_signed,
            input_acc_rounding_mode_s=input_acc_rounding_mode,
            forget_acc_narrow_i=forget_acc_narrow_range,
            forget_acc_signed_i=forget_acc_signed,
            forget_acc_rounding_mode_s=forget_acc_rounding_mode,
            cell_acc_narrow_i=cell_acc_narrow_range,
            cell_acc_signed_i=cell_acc_signed,
            cell_acc_rounding_mode_s=cell_acc_rounding_mode,
            output_acc_narrow_i=output_acc_narrow_range,
            output_acc_signed_i=output_acc_signed,
            output_acc_rounding_mode_s=output_acc_rounding_mode,
            input_sigmoid_narrow_i=input_sigmoid_narrow_range,
            input_sigmoid_signed_i=input_sigmoid_signed,
            input_sigmoid_rounding_mode_s=input_sigmoid_rounding_mode,
            forget_sigmoid_narrow_i=forget_sigmoid_narrow_range,
            forget_sigmoid_signed_i=forget_sigmoid_signed,
            forget_sigmoid_rounding_mode_s=forget_sigmoid_rounding_mode,
            cell_tanh_narrow_i=cell_tanh_narrow_range,
            cell_tanh_signed_i=cell_tanh_signed,
            cell_tanh_rounding_mode_s=cell_tanh_rounding_mode,
            output_sigmoid_narrow_range_i=output_sigmoid_narrow_range,
            output_sigmoid_signed_i=output_sigmoid_signed,
            output_sigmoid_rounding_mode_s=output_sigmoid_rounding_mode,
            hidden_state_tanh_narrow_i=hidden_state_tanh_narrow_range,
            hidden_state_tanh_signed_i=hidden_state_tanh_signed,
            hidden_state_tanh_rounding_mode_s=hidden_state_tanh_rounding_mode,
            # PyTorch requires to specify the number of outputs manually
            outputs=3,
        )

    @staticmethod
    def forward(
        ctx,  # args and kwargs passed from _QuantLSTMLayer
        quant_input,
        quant_hidden_state,
        quant_cell_state,
        quant_weight_ii,
        quant_weight_if,
        quant_weight_ic,
        quant_weight_io,
        quant_weight_hi,
        quant_weight_hf,
        quant_weight_hc,
        quant_weight_ho,
        quant_bias_input,
        quant_bias_forget,
        quant_bias_cell,
        quant_bias_output,  # Symbolic kwargs passed from BrevitasQuantLSTMLayerHandler
        batch_first,
        reverse_input,
        cifg,  # Output quant
        output_scale,
        output_zero_point,
        output_bit_width,
        output_narrow_range,
        output_signed,
        output_rounding_mode,  # Cell state quant
        cell_state_scale,
        cell_state_zero_point,
        cell_state_bit_width,
        cell_state_narrow_range,
        cell_state_signed,
        cell_state_rounding_mode,  # Input gate accumulator quant
        input_acc_scale,
        input_acc_zero_point,
        input_acc_bit_width,
        input_acc_narrow_range,
        input_acc_signed,
        input_acc_rounding_mode,  # Forget gate accumulator quant
        forget_acc_scale,
        forget_acc_zero_point,
        forget_acc_bit_width,
        forget_acc_narrow_range,
        forget_acc_signed,
        forget_acc_rounding_mode,  # Cell gate accumulator quant
        cell_acc_scale,
        cell_acc_zero_point,
        cell_acc_bit_width,
        cell_acc_narrow_range,
        cell_acc_signed,
        cell_acc_rounding_mode,  # Output gate accumulator quant
        output_acc_scale,
        output_acc_zero_point,
        output_acc_bit_width,
        output_acc_narrow_range,
        output_acc_signed,
        output_acc_rounding_mode,  # Input gate sigmoid quant
        input_sigmoid_scale,
        input_sigmoid_zero_point,
        input_sigmoid_bit_width,
        input_sigmoid_narrow_range,
        input_sigmoid_signed,
        input_sigmoid_rounding_mode,  # Forget gate sigmoid quant
        forget_sigmoid_scale,
        forget_sigmoid_zero_point,
        forget_sigmoid_bit_width,
        forget_sigmoid_narrow_range,
        forget_sigmoid_signed,
        forget_sigmoid_rounding_mode,  # Cell gate tanh quant
        cell_tanh_scale,
        cell_tanh_zero_point,
        cell_tanh_bit_width,
        cell_tanh_narrow_range,
        cell_tanh_signed,
        cell_tanh_rounding_mode,  # Output gate sigmoid quant
        output_sigmoid_scale,
        output_sigmoid_zero_point,
        output_sigmoid_bit_width,
        output_sigmoid_narrow_range,
        output_sigmoid_signed,
        output_sigmoid_rounding_mode,  # Hidden state tanh quant
        hidden_state_tanh_scale,
        hidden_state_tanh_zero_point,
        hidden_state_tanh_bit_width,
        hidden_state_tanh_narrow_range,
        hidden_state_tanh_signed,
        hidden_state_tanh_rounding_mode,
    ):
        # Tp simplify things, here we are returning the outputs
        # as if they were already concatenated. Scale/zp/bw are avoided too.
        # This preserves output shapes but not values.
        # See _QuantLSTMCell for the actual implementation.
        quant_outputs = torch.zeros(
            quant_input.size(0), quant_input.size(1), quant_hidden_state.size(1), device=quant_hidden_state.device
        )
        return quant_outputs, quant_hidden_state, quant_cell_state
