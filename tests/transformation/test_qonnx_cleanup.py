import os
import urllib.request

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup


def test_cleanup_cnv_w2a2():
    # download test data
    dl_dir = "/tmp"
    dl_file = dl_dir + "/cnv-w2a2.onnx"
    cnv_w2a2_qonnx_url = (
        "https://raw.githubusercontent.com/fastmachinelearning/"
        "QONNX_model_zoo/main/models/CIFAR10/Brevitas_FINN_CNV/CNV_2W2A.onnx"
    )
    urllib.request.urlretrieve(cnv_w2a2_qonnx_url, dl_file)
    assert os.path.isfile(dl_file)
    # run cleanup with default settings
    out_file = dl_dir + "/cnv-w2a2-clean.onnx"
    cleanup(dl_file, out_file=out_file)
    assert os.path.isfile(out_file)
    model = ModelWrapper(out_file)
    # check some names and shapes
    assert model.check_all_tensor_shapes_specified()
    assert model.graph.output[0].name == "global_out"
    assert model.get_tensor_shape(model.graph.output[0].name) == [1, 10]
    # constant folding should have replaced Shape -> Gather -> Unsqueeze -> Concat -> Reshape
    # with a Reshape w/ shape=(1,-1)
    reshape_nodes = model.get_nodes_by_op_type("Reshape")
    assert len(reshape_nodes) == 1
    reshape_node = reshape_nodes[0]
    assert (model.get_initializer(reshape_node.input[1]) == [1, -1]).all()
    os.remove(dl_file)
