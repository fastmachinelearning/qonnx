# Copyright (c) 2026 Advanced Micro Devices, Inc.
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
# * Neither the name of Advanced Micro Devices, Inc. nor the names of its
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
import onnx.parser as oprs
from onnx.checker import check_model

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import FillEmptyRoI
from qonnx.transformation.infer_shapes import InferShapes


@pytest.mark.parametrize("scale_factor", [2, 4])
def test_fill_empty_roi(scale_factor):
    """Test that FillEmptyRoI handles Resize nodes with empty RoI string."""
    ifm = 10
    ich = 3
    ishp = (1, ich, ifm, ifm)
    oshp = (1, ich, ifm * scale_factor, ifm * scale_factor)
    rscales = np.array([1.0, 1.0, float(scale_factor), float(scale_factor)], dtype=np.float32)

    ishp_str = str(list(ishp))
    oshp_str = str(list(oshp))

    # Create a model with Resize node that has empty string for RoI
    input_str = f"""
    <
        ir_version: 7,
        opset_import: ["" : 13]
    >
    agraph (float{ishp_str} in0) => (float{oshp_str} out0)
    <
        float scales
    >
    {{
        out0 = Resize<
            mode="nearest"
        >(in0, , scales)
    }}
    """

    model = oprs.parse_model(input_str)
    model = ModelWrapper(model)
    model.set_initializer("scales", rscales)

    resize_nodes = [n for n in model.graph.node if n.op_type == "Resize"]
    assert len(resize_nodes) == 1, "Should have exactly one Resize node"

    resize_node = resize_nodes[0]
    assert len(resize_node.input) >= 2, "Resize should have at least 2 inputs"
    assert resize_node.input[1] == "", "RoI input should be empty string before transformation"

    model_transformed = model.transform(FillEmptyRoI())
    model_transformed = model_transformed.transform(InferShapes())

    resize_nodes = [n for n in model_transformed.graph.node if n.op_type == "Resize"]
    assert len(resize_nodes) == 1

    resize_node = resize_nodes[0]
    assert resize_node.input[1] != "", "RoI input should not be empty after transformation"

    roi_name = resize_node.input[1]
    roi_init = model_transformed.get_initializer(roi_name)
    assert roi_init is not None, f"RoI initializer {roi_name} should exist"
    assert roi_init.shape == (0,), f"RoI tensor should have shape (0,), got {roi_init.shape}"

    check_model(model_transformed.model)


def test_fill_empty_roi_doesnt_modify_valid():
    """Verify transformation doesn't modify resize nodes that already have valid ROI."""
    ifm = 10
    ich = 3
    scale_factor = 2
    ishp = (1, ich, ifm, ifm)
    oshp = (1, ich, ifm * scale_factor, ifm * scale_factor)
    rscales = np.array([1.0, 1.0, float(scale_factor), float(scale_factor)], dtype=np.float32)

    ishp_str = str(list(ishp))
    oshp_str = str(list(oshp))

    # Create model with valid ROI.
    input_str = f"""
    <
        ir_version: 7,
        opset_import: ["" : 13]
    >
    agraph (float{ishp_str} in0) => (float{oshp_str} out0)
    <
        float roi,
        float scales
    >
    {{
        out0 = Resize<mode="nearest">(in0, roi, scales)
    }}
    """

    model = oprs.parse_model(input_str)
    model = ModelWrapper(model)
    model.set_initializer("roi", np.empty([0], dtype=np.float32))
    model.set_initializer("scales", rscales)

    resize_nodes_before = [n for n in model.graph.node if n.op_type == "Resize"]
    roi_name_before = resize_nodes_before[0].input[1]

    model_transformed = model.transform(FillEmptyRoI())

    resize_nodes_after = [n for n in model_transformed.graph.node if n.op_type == "Resize"]
    roi_name_after = resize_nodes_after[0].input[1]

    assert roi_name_before == roi_name_after, "Should not modify Resize with valid RoI"
    assert len(model.graph.initializer) == len(model_transformed.graph.initializer), "Should not add extra initializers"


def test_fill_empty_roi_idempotent():
    """Verify transformation is idempotent."""
    ifm = 10
    ich = 3
    scale_factor = 2
    ishp = (1, ich, ifm, ifm)
    oshp = (1, ich, ifm * scale_factor, ifm * scale_factor)
    rscales = np.array([1.0, 1.0, float(scale_factor), float(scale_factor)], dtype=np.float32)

    ishp_str = str(list(ishp))
    oshp_str = str(list(oshp))

    # Create model with empty ROI
    input_str = f"""
    <
        ir_version: 7,
        opset_import: ["" : 13]
    >
    agraph (float{ishp_str} in0) => (float{oshp_str} out0)
    <
        float scales
    >
    {{
        out0 = Resize<mode="nearest">(in0, , scales)
    }}
    """

    model = oprs.parse_model(input_str)
    model = ModelWrapper(model)
    model.set_initializer("scales", rscales)

    # Apply transformation twice
    model = model.transform(FillEmptyRoI())
    roi_name_first = [n for n in model.graph.node if n.op_type == "Resize"][0].input[1]
    num_initializers_first = len(model.graph.initializer)

    model = model.transform(FillEmptyRoI())
    roi_name_second = [n for n in model.graph.node if n.op_type == "Resize"][0].input[1]
    num_initializers_second = len(model.graph.initializer)

    # Verify idempotency
    assert roi_name_first == roi_name_second, "Should not change RoI on second run"
    assert num_initializers_first == num_initializers_second, "Should not add duplicate initializers"
