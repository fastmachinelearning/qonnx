# Copyright (c) 2022 Advanced Micro Devices, Inc.
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

import clize
import json

import qonnx.analysis.inference_cost as infca
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes


def compute_bops(inf_cost_dict):
    total_bops = 0.0
    for k, v in inf_cost_dict.items():
        if k.startswith("op_mac"):
            comps = k.split("_")
            dt1 = DataType[comps[2]]
            dt2 = DataType[comps[3]]
            total_bops += dt1.bitwidth() * dt2.bitwidth() * v
    return total_bops


def compute_mem_bits(inf_cost_dict, filter_string="mem_w"):
    total_mem_bits = 0.0
    for k, v in inf_cost_dict.items():
        if k.startswith(filter_string):
            comps = k.split("_")
            dt = DataType[comps[2]]
            total_mem_bits += dt.bitwidth() * v
    return total_mem_bits


def inference_cost(
    model_filename_or_wrapper, *, output_json=None, output_onnx=None, preprocess=True, discount_sparsity=True
):
    """Return the inference cost estimate metric for given ONNX model.
    Supports the Quant op for weight/activation quantization.

    :param model_filename_or_wrapper: Filename or ModelWrapper for ONNX model
    :param output_json: Optional JSON filename to save the inference cost dict
    :param output_onnx: Optional ONNX filename to save the final model after any
        preprocessing
    :param preprocess: If set, run preprocessing steps such as shape inference,
        datatype inference and constant folding. Strongly recommended.
    :param discount_sparsity: If set, will discount op cost of MAC ops with a
        constant zero weight, and the mem cost of constant zero weights.
    """
    if isinstance(model_filename_or_wrapper, ModelWrapper):
        model = model_filename_or_wrapper
    else:
        model = ModelWrapper(model_filename_or_wrapper)
    if preprocess:
        qnt_nodes = model.get_nodes_by_op_type("Quant")
        for qnt_node in qnt_nodes:
            qnt_node.domain = "qonnx.custom_op.general"
        model = model.transform(InferShapes())
        model = model.transform(GiveUniqueParameterTensors())
        model = model.transform(InferDataTypes())
        model = model.transform(FoldConstants(exclude_op_types=[]))
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(RemoveStaticGraphInputs())
        model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    if output_onnx is not None:
        model.save(output_onnx)
    ret = model.analysis(lambda x: infca.inference_cost(x, discount_sparsity))
    bops = compute_bops(ret)
    mem_w_bits = compute_mem_bits(ret, "mem_w")
    mem_o_bits = compute_mem_bits(ret, "mem_o")
    ret["total_bops"] = bops
    ret["total_mem_w_bits"] = mem_w_bits
    ret["total_mem_o_bits"] = mem_o_bits

    if "unsupported" in ret:
        ret["unsupported"] = str(ret["unsupported"])

    if output_json is not None:
        with open(output_json, "w") as f:
            json.dump(ret, f, sort_keys=True, indent=2)

    return ret


def main():
    clize.run(inference_cost)


if __name__ == "__main__":
    main()
