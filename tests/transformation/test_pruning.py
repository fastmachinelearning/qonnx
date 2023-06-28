# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

from qonnx.core.datatype import DataType
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.pruning import (
    ApplyMasks,
    PropagateMasks,
    PruneChannels,
    RemoveMaskedChannels,
    remove_masked_tensor_channels,
)
from qonnx.util.cleanup import cleanup_model
from qonnx.util.inference_cost import inference_cost
from qonnx.util.test import download_model, get_golden_in_and_output


def test_remove_masked_tensor_channels():
    shp = (4, 5, 6)
    x = np.random.rand(*shp)
    assert remove_masked_tensor_channels(x, [0, 2], axis=0).shape == (2, 5, 6)
    assert remove_masked_tensor_channels(x, [3], axis=1).shape == (4, 4, 6)
    assert remove_masked_tensor_channels(shp, [3], axis=1) == (4, 4, 6)


def test_apply_and_propagate_masks():
    model = download_model("FINN-TFC_W2A2", do_cleanup=True, return_modelwrapper=True)
    # manifest quantized weights as initializers
    model = model.transform(FoldConstants([]))
    mm_nodes = model.get_nodes_by_op_type("MatMul")
    # mark channels 0 and 3 from tensor Mul_0_out0
    # and channel 6 for the input to the 2nd MatMul as well as
    # channel 2 of input and 5 of output from the matmul weight
    # to be pruned
    prune_spec = {"Mul_0_out0": {0, 3}, mm_nodes[0].input[1]: {"i2", "o5"}, mm_nodes[1].input[0]: {6}}
    model = model.transform(ApplyMasks(prune_spec))
    assert model.get_tensor_sparsity("Mul_0_out0") == prune_spec["Mul_0_out0"]
    assert model.get_tensor_sparsity(mm_nodes[1].input[0]) == prune_spec[mm_nodes[1].input[0]]
    assert model.get_tensor_sparsity(mm_nodes[0].input[1]) == prune_spec[mm_nodes[0].input[1]]
    # now apply the propagation
    model = model.transform(PropagateMasks())
    assert model.get_tensor_sparsity("Mul_0_out0") == {0, 2, 3}
    assert model.get_tensor_sparsity(mm_nodes[0].input[1]) == {"i0", "i2", "i3", "o5", "o6"}
    assert model.get_tensor_sparsity("BatchNormalization_0_out0") == {5, 6}
    model = model.transform(RemoveMaskedChannels())
    assert tuple(model.get_tensor_shape(mm_nodes[0].input[0])) == (1, 781)
    assert tuple(model.get_tensor_shape(mm_nodes[0].input[1])) == (781, 62)
    assert tuple(model.get_tensor_shape(mm_nodes[1].input[0])) == (1, 62)
    assert tuple(model.get_tensor_shape(mm_nodes[1].input[1])) == (62, 64)


def test_pruning_mnv1():
    model = download_model("MobileNetv1-w4a4", return_modelwrapper=True)
    # mark input as scaled 8-bit to get correct inference cost
    model.set_tensor_datatype(model.graph.input[0].name, DataType["SCALEDINT<8>"])
    model = model.transform(InferDataTypes())
    # do cleanup including folding quantized weights
    model = cleanup_model(model, False)
    inp, golden = get_golden_in_and_output("MobileNetv1-w4a4")
    cost0 = inference_cost(model, discount_sparsity=False)
    assert cost0["op_mac_SCALEDINT<8>_SCALEDINT<8>"] == 10645344.0
    assert cost0["mem_w_SCALEDINT<8>"] == 864.0
    assert cost0["op_mac_SCALEDINT<4>_SCALEDINT<4>"] == 556357408.0
    assert cost0["mem_w_SCALEDINT<4>"] == 4208224.0
    prune_spec = {
        "Quant_0_out0": {4, 6, 10, 13, 15, 16, 19, 26, 28},
        "Quant_1_out0": {0, 4, 6, 10, 15, 19, 26, 28},
        "Quant_2_out0": {42},
        "Quant_3_out0": {42},
        "Quant_6_out0": {102},
        "Quant_7_out0": {102},
    }

    model = model.transform(PruneChannels(prune_spec))
    cost1 = inference_cost(model, discount_sparsity=False)
    assert cost1["op_mac_SCALEDINT<8>_SCALEDINT<8>"] == 7318674.0
    assert cost1["mem_w_SCALEDINT<8>"] == 594.0
    assert cost1["op_mac_SCALEDINT<4>_SCALEDINT<4>"] == 546053216.0
    assert cost1["mem_w_SCALEDINT<4>"] == 4206942.0
    iname = model.graph.input[0].name
    oname = model.graph.output[0].name
    ret = execute_onnx(model, {iname: inp})[oname]
    assert np.isclose(golden, ret).all()
