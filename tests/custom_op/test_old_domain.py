import sys
from onnx import helper, TensorProto

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model


def test_get_custom_op_old_domain():
    assert "legacy_custom_op" not in sys.modules

    node = helper.make_node(
        "LegacyAdd",
        ["a", "b"],
        ["c"],
        domain="legacy_custom_op",
    )

    graph = helper.make_graph(
        [node],
        "legacy_graph",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [1])],
    )
    model = qonnx_make_model(graph, producer_name="legacy-test")
    model = ModelWrapper(model)

    inst = getCustomOp(model.graph.node[0])
    assert inst.__class__.__name__ == "LegacyAdd"
