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

import numpy as np
import onnx
import onnx.numpy_helper as np_helper
import os
import urllib.request as ureq
from pkgutil import get_data

import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes


def test_renaming():
    # load the onnx model
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    # do some basic checks
    assert model.graph.input[0].name == "global_in"
    assert model.graph.output[0].name == "global_out"
    assert model.graph.node[1].op_type == "Conv"
    assert model.graph.node[1].name == "Conv_0"
    assert model.graph.node[1].input[1] == "Conv_0_param0"
    assert model.graph.node[6].op_type == "Add"
    assert model.graph.node[6].name == "Add_1"
    assert model.graph.node[6].input[1] == "Add_1_param0"
    # ensure running renaming twice still yields the same names
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    assert model.graph.node[1].op_type == "Conv"
    assert model.graph.node[1].name == "Conv_0"
    assert model.graph.node[1].input[1] == "Conv_0_param0"
    assert model.graph.node[6].op_type == "Add"
    assert model.graph.node[6].name == "Add_1"
    assert model.graph.node[6].input[1] == "Add_1_param0"
    # run renamed model to make sure we did not mess up the topology
    raw_i = get_data("qonnx.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
    raw_o = get_data("qonnx.data", "onnx/mnist-conv/test_data_set_0/output_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    output_tensor = onnx.load_tensor_from_string(raw_o)
    input_dict = {"global_in": np_helper.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model, input_dict)
    assert np.isclose(np_helper.to_array(output_tensor), output_dict["global_out"], atol=1e-3).all()


def test_rename_multi_io_tinyyolov3():
    download_url = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation"
    download_url += "/tiny-yolov3/model/tiny-yolov3-11.onnx"
    export_onnx_path = download_url.split("/")[-1]
    ureq.urlretrieve(download_url, export_onnx_path)
    if not os.path.isfile(export_onnx_path):
        pytest.skip("Couldn't download ONNX model, skipping")
    model = ModelWrapper(export_onnx_path)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    assert len(model.graph.input) == 2
    assert model.graph.input[0].name == "global_in"
    assert model.graph.input[1].name == "global_in_1"
    assert len(model.graph.output) == 3
    assert model.graph.output[0].name == "global_out"
    assert model.graph.output[1].name == "global_out_1"
    assert model.graph.output[2].name == "global_out_2"
    os.remove(export_onnx_path)
