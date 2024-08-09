# Copyright (c) 2021 Xilinx, Inc.
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
#   and/or other materials provided with the distributionode.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permissionode.
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

from qonnx.util.basic import get_by_name


def get_node_tensor_dtypes(model, node):
    # input tensor (input 0)
    i_name = node.input[0]
    i_dtype = model.get_tensor_datatype(i_name)
    # weight tensor (input 1)
    w_name = node.input[1]
    w_dtype = model.get_tensor_datatype(w_name)
    # output tensor (input 0)
    o_name = node.output[0]
    o_dtype = model.get_tensor_datatype(o_name)
    return (i_dtype, w_dtype, o_dtype)


def get_node_tensor_shapes(model, node):
    # input tensor (input 0)
    i_name = node.input[0]
    i_shape = model.get_tensor_shape(i_name)
    assert i_shape is not None, "Input has undefined shape: " + str(node)
    # weight tensor (input 1)
    w_name = node.input[1]
    w_shape = model.get_tensor_shape(w_name)
    assert w_shape is not None, "Weight has undefined shape: " + str(node)
    # output tensor (output 0)
    o_name = node.output[0]
    o_shape = model.get_tensor_shape(o_name)
    assert o_shape is not None, "Output has undefined shape: " + str(node)
    return (i_shape, w_shape, o_shape)


def get_node_weight_density(model, w_name):
    w_tensor = model.get_initializer(w_name)
    if w_tensor is None:
        return 1.0
    w_total = np.prod(w_tensor.shape)
    w_density = np.count_nonzero(w_tensor) / w_total
    return w_density


def aggregate_dict_keys(res_dict):
    total_dict = {}
    for layer in res_dict:
        layer_res_dict = res_dict[layer]
        for r_type in layer_res_dict.keys():
            if "efficiency" in r_type:
                continue
            r_amount = layer_res_dict[r_type]
            r_amount = float(r_amount)
            if r_type in total_dict.keys():
                total_dict[r_type] += r_amount
            else:
                total_dict[r_type] = r_amount
    return total_dict


def inference_cost_conv(model, node, discount_sparsity):
    # extract info about the conv kernel attributes
    k = get_by_name(node.attribute, "kernel_shape").ints
    k_prod = np.prod(k)
    group = get_by_name(node.attribute, "group")
    if group is None:
        group = 1
    else:
        group = group.i
    # extract info from tensor shapes and datatypes
    (i_dtype, w_dtype, o_dtype) = get_node_tensor_dtypes(model, node)
    (i_shape, w_shape, o_shape) = get_node_tensor_shapes(model, node)
    bsize = i_shape[0]
    ifm_ch = i_shape[1]
    ofm_ch = o_shape[1]
    assert ofm_ch == w_shape[0], "Mismatch in output channels"
    assert ofm_ch % group == 0, "Invalid group setting: " + str(node)
    ofm_pix_total = np.prod(o_shape[2:])
    n_macs = bsize * (ofm_ch // group) * ifm_ch * k_prod * ofm_pix_total
    w_mem = np.prod(w_shape)
    o_mem = np.prod(o_shape)
    if discount_sparsity:
        wname = node.input[1]
        density = get_node_weight_density(model, wname)
        n_macs *= density
        w_mem *= density
    idt_name = i_dtype.name
    wdt_name = w_dtype.name
    odt_name = o_dtype.name
    mac_op_type_str = "op_mac_%s_%s" % (idt_name, wdt_name)
    w_mem_type_str = "mem_w_%s" % (wdt_name)
    o_mem_type_str = "mem_o_%s" % (odt_name)
    # keep in floats to remain compatible with json serialization
    n_macs, w_mem, o_mem = float(n_macs), float(w_mem), float(o_mem)
    ret = {mac_op_type_str: n_macs, w_mem_type_str: w_mem, o_mem_type_str: o_mem}
    return ret


def inference_cost_matmul(model, node, discount_sparsity):
    # extract info from tensor shapes and datatypes
    (i_dtype, w_dtype, o_dtype) = get_node_tensor_dtypes(model, node)
    (i_shape, w_shape, o_shape) = get_node_tensor_shapes(model, node)
    if node.op_type == "Gemm":
        assert len(i_shape) == 2 and len(w_shape) == 2
        tA = get_by_name(node.attribute, "transA")
        tB = get_by_name(node.attribute, "transB")
        if tA is not None and tA.i == 1:
            i_shape = i_shape[::-1]
        if tB is not None and tB.i == 1:
            w_shape = w_shape[::-1]
    # exclude common dim (last axis) from one side to avoid duplication
    n_macs = i_shape[-1] * np.prod(o_shape)
    # deal with both dyn,param and dyn,dyn cases for weight memory
    inp0_is_const = model.get_initializer(node.input[0]) is not None
    inp1_is_const = model.get_initializer(node.input[1]) is not None
    if inp0_is_const and (not inp1_is_const):
        # inp 0 is static
        w_mem = np.prod(i_shape)
        wname = node.input[0]
    elif (not inp0_is_const) and inp1_is_const:
        # inp 1 is static
        w_mem = np.prod(w_shape)
        wname = node.input[1]
    elif (not inp0_is_const) and (not inp1_is_const):
        # both inputs dynamic
        w_mem = 0
        wname = None
    if discount_sparsity and wname is not None:
        density = get_node_weight_density(model, wname)
        n_macs *= density
        w_mem *= density
    o_mem = np.prod(o_shape)
    idt_name = i_dtype.name
    wdt_name = w_dtype.name
    odt_name = o_dtype.name
    mac_op_type_str = "op_mac_%s_%s" % (idt_name, wdt_name)
    w_mem_type_str = "mem_w_%s" % (wdt_name)
    o_mem_type_str = "mem_o_%s" % (odt_name)
    # keep in floats to remain compatible with json serialization
    n_macs, w_mem, o_mem = float(n_macs), float(w_mem), float(o_mem)
    ret = {mac_op_type_str: n_macs, w_mem_type_str: w_mem, o_mem_type_str: o_mem}
    return ret


def inference_cost_upsample(model, node, discount_sparsity):
    # extract info about the upsampling kernel attributes
    mode = get_by_name(node.attribute, "mode").s.decode("utf-8")
    scales_tensor = node.input[1]
    scales_initializer = model.get_initializer(scales_tensor)

    # extract info from tensor shapes and datatypes
    (i_dtype, scale_dtype, o_dtype) = get_node_tensor_dtypes(model, node)
    (i_shape, scale_shape, o_shape) = get_node_tensor_shapes(model, node)
    bsize = i_shape[0]
    ifm_ch = i_shape[1]
    ofm_pix_total = np.prod(o_shape[2:])

    # MAC calculation
    if mode == "nearest":
        # No calculation involved, since data is just copied over multiple times
        n_macs = 0
    elif mode == "linear":
        # Data gets linearly interpolated in each dimension
        # Two MACs per dimension and output pixel assumed
        n_dim_scaling = np.sum(scales_initializer > 1)
        n_macs = 2 * n_dim_scaling * ofm_pix_total * ifm_ch * bsize
    else:
        raise ValueError(f"Upsampling mode {mode} not supported for estimation.")

    # Mem calculation
    o_mem = np.prod(o_shape)
    idt_name = i_dtype.name
    odt_name = o_dtype.name
    mac_op_type_str = "op_mac_%s_%s" % (idt_name, idt_name)
    o_mem_type_str = "mem_o_%s" % (odt_name)

    # keep in floats to remain compatible with json serialization
    n_macs, o_mem = float(n_macs), float(o_mem)
    ret = {mac_op_type_str: n_macs, o_mem_type_str: o_mem}
    return ret


def inference_cost(model, discount_sparsity=True, cost_breakdown=False):
    "Ensure all nodes have unique names prior to calling this analysis pass."

    ret, node_costs, nodes_per_optype = {}, {}, {}
    zero_cost_ops = [
        "MaxPool",
        "AveragePool",
        "Quant",
        "QuantizeLinear",
        "DequantizeLinear",
        "Reshape",
        "Concat",
        "Transpose",
        "Div",
        "Mul",
        "Add",
        "Sub",
        "BatchNormalization",
        "Relu",
        "Elu",
        "Selu",
        "Sigmoid",
        "Identity",
        "Flatten",
        "Pad",
        "Clip",
        "Trunc",
    ]
    unsupported_ops = set()
    inference_cost_fxn_map = {
        "Conv": inference_cost_conv,
        "MatMul": inference_cost_matmul,
        "Gemm": inference_cost_matmul,
        "Upsample": inference_cost_upsample,
    }
    for node in model.graph.node:
        if node.op_type in inference_cost_fxn_map.keys():
            node_cost = inference_cost_fxn_map[node.op_type](model, node, discount_sparsity)
            node_costs[node.name] = node_cost
            if node.op_type not in nodes_per_optype.keys():
                new_optype = {}
                new_optype[node.name] = node_cost
                nodes_per_optype[node.op_type] = new_optype
            else:
                nodes_per_optype[node.op_type][node.name] = node_cost
        elif node.op_type in zero_cost_ops:
            continue
        else:
            unsupported_ops.add(node.op_type)
    total = aggregate_dict_keys(node_costs)
    total["unsupported"] = unsupported_ops
    total["discount_sparsity"] = discount_sparsity
    ret["total_cost"] = total
    if cost_breakdown:
        optype_cost = {}
        for optype, resources in nodes_per_optype.items():
            optype_cost[optype] = aggregate_dict_keys(resources)
        ret["optype_cost"] = optype_cost
        ret["node_cost"] = node_costs
    return ret
