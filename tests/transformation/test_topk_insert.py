import pytest

import numpy as np
import onnx
import onnx.numpy_helper as nph
from pkgutil import get_data

import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK


@pytest.mark.parametrize("k", [1, 2])
def test_topk_insert(k):
    raw_m = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model.model.opset_import[0].version = 11

    # do transformations (no topk)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())

    # verification: generate random input, run through net, streamline,
    # run again, check that output is top-k
    raw_i = get_data("qonnx.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    input_tensor = nph.to_array(input_tensor)
    input_dict = {"global_in": input_tensor}
    output_golden = oxe.execute_onnx(model, input_dict)["global_out"]
    output_golden_topk = np.flip(output_golden.flatten().argsort())[:k]
    output_golden_topk = output_golden_topk.flatten()

    # insert top-k
    model = model.transform(InsertTopK(k))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferShapes())

    # verify output of top-k
    output_dict_topk = oxe.execute_onnx(model, input_dict)
    output_pysim_topk = output_dict_topk[list(output_dict_topk.keys())[0]]
    output_pysim_topk = output_pysim_topk.astype(np.int32).flatten()

    assert np.array_equal(output_golden_topk, output_pysim_topk)
