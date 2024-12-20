# Copyright (c) 2024 Advanced Micro Devices, Inc.
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

import os
import random
import urllib.request

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.quantize_graph import QuantizeGraph
from qonnx.util.cleanup import cleanup
from qonnx.util.inference_cost import inference_cost

random.seed(42)

download_url = "https://github.com/onnx/models/raw/main/validated/vision/"
download_url += "classification/resnet/model/resnet18-v1-7.onnx?download="

model_details = {
    "resnet18-v1-7": {
        "description": "Resnet18 Opset version 7.",
        "url": download_url,
        "test_input": {
            "name": {
                "Conv_0": [
                    (("input", 0), (1, 0, 8, 0, 1, "ROUND")),
                    (("input", 1), (1, 0, 8, 0, 1, "ROUND")),
                    (("output", 0), (1, 0, 8, 0, 1, "ROUND")),
                ],
                "Conv_1": [(("input", 0), (1, 0, 8, 0, 1, "ROUND"))],
                "Conv_2": [(("input", 1), (1, 0, 8, 0, 1, "ROUND")), (("output", 0), (1, 0, 8, 0, 1, "ROUND"))],
            },
            "op_type": {
                "Gemm": [
                    (("input", 0), (1, 0, 8, 0, 1, "ROUND")),
                    (("input", 1), (1, 0, 8, 0, 1, "ROUND")),
                    (("input", 2), (1, 0, 8, 0, 1, "ROUND")),
                    (("output", 0), (1, 0, 8, 0, 1, "ROUND")),
                ]
            },
        },
    },
}


def download_model(test_model, do_cleanup=False, return_modelwrapper=False):
    qonnx_url = model_details[test_model]["url"]
    # download test data
    dl_dir = "/tmp"
    dl_file = dl_dir + f"/{test_model}.onnx"
    ret = dl_file
    if not os.path.isfile(dl_file):
        urllib.request.urlretrieve(qonnx_url, dl_file)
    if do_cleanup:
        out_file = dl_dir + f"/{test_model}_clean.onnx"
        cleanup(dl_file, out_file=out_file, override_inpsize=1)
        ret = out_file
    if return_modelwrapper:
        ret = ModelWrapper(ret)
    return ret


def to_verify(model, test_details):
    by = random.choice(list(test_details.keys()))  # by "name" or "op_type"

    if by == "name":
        sample_node_name = random.choice(list(test_details["name"].keys()))
        sample_node = model.get_node_from_name(sample_node_name)
        sample_pos = random.choice(test_details["name"][sample_node_name])
    if by == "op_type":
        node_type = random.choice(list(test_details["op_type"].keys()))
        sample_node = random.choice(model.get_nodes_by_op_type(node_type))
        sample_pos = random.choice(test_details["op_type"][node_type])

    if sample_pos[0][0] == "input":
        tensor_to_verify = sample_node.input[sample_pos[0][1]]
        producer_node = model.find_producer(tensor_to_verify)
        if producer_node.op_type == "Quant":
            verification = "Success"
        else:
            verification = "Failure"
    if sample_pos[0][0] == "output":
        tensor_to_verify = sample_node.output[sample_pos[0][1]]
        consumer_node = model.find_consumer(tensor_to_verify)
        if consumer_node.op_type == "Quant":
            verification = "Success"
        else:
            verification = "Failure"

    return verification


@pytest.mark.parametrize("test_model", model_details.keys())
def test_quantize_graph(test_model):
    test_details = model_details[test_model]
    model = download_model(test_model, do_cleanup=True, return_modelwrapper=True)
    original_model_inf_cost = inference_cost(model, discount_sparsity=False)["total_cost"]
    nodes_pos = test_details["test_input"]
    model = model.transform(QuantizeGraph(nodes_pos))
    quantnodes_added = len(model.get_nodes_by_op_type("Quant"))
    assert quantnodes_added == 10  # 10 positions are specified.
    verification = to_verify(model, nodes_pos)
    assert verification == "Success"
    inf_cost = inference_cost(model, discount_sparsity=False)["total_cost"]
    assert (
        inf_cost["total_macs"] == original_model_inf_cost["total_macs"]
    )  # "1814073344.0" must be same as the original model.
    assert (
        inf_cost["total_mem_w_elems"] == original_model_inf_cost["total_mem_w_elems"]
    )  # "11678912.0" must be same as the original model.
    assert (
        inf_cost["total_mem_o_bits"] == original_model_inf_cost["total_mem_o_bits"]
    )  # "79510784.0" must be same as the original model.
    assert (
        inf_cost["total_mem_o_elems"] == original_model_inf_cost["total_mem_o_elems"]
    )  # "2484712.0" must be same as the original model.
    assert inf_cost["total_bops"] == 1566256136192.0
    assert inf_cost["total_mem_w_bits"] == 360326656.0
    assert inf_cost["op_mac_INT8_INT8"] == 118525952.0
