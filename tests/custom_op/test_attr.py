import numpy as np
import onnx.parser as oprs

import qonnx.custom_op.general as general
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import getCustomOp


class AttrTestOp(CustomOp):
    def get_nodeattr_types(self):
        return {"tensor_attr": ("t", True, np.asarray([]))}

    def make_shape_compatible_op(self, model):
        param_tensor = self.get_nodeattr("tensor_attr")
        return super().make_const_shape_op(param_tensor.shape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        param_tensor = self.get_nodeattr("tensor_attr")
        context[node.output[0]] = param_tensor

    def verify_node(self):
        pass


def test_attr():
    general.custom_op["AttrTestOp"] = AttrTestOp
    ishp = (1, 10)
    wshp = (1, 3)
    oshp = wshp
    ishp_str = str(list(ishp))
    oshp_str = str(list(oshp))
    wshp_str = str(list(wshp))
    w = np.asarray([1, -2, 3], dtype=np.int8)
    strarr = np.array2string(w, separator=", ")
    w_str = strarr.replace("[", "{").replace("]", "}").replace(" ", "")
    tensor_attr_str = f"int8{wshp_str} {w_str}"

    input = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float{ishp_str} in0) => (int8{oshp_str} out0)
    {{
        out0 = qonnx.custom_op.general.AttrTestOp<
            tensor_attr={tensor_attr_str}
        >(in0)
    }}
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    inst = getCustomOp(model.graph.node[0])
    w_prod = inst.get_nodeattr("tensor_attr")
    assert (w_prod == w).all()
    w = w - 1
    inst.set_nodeattr("tensor_attr", w)
    w_prod = inst.get_nodeattr("tensor_attr")
    assert (w_prod == w).all()
