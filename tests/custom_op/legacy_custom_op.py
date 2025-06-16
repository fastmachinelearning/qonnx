from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import register_op

@register_op(domain="legacy_custom_op", op_type="LegacyAdd")
class LegacyAdd(CustomOp):
    def get_nodeattr_types(self):
        return {}

    def make_shape_compatible_op(self, model):
        return super().make_const_shape_op([1])

    def infer_node_datatype(self, model):
        pass

    def execute_node(self, context, graph):
        a = context[self.onnx_node.input[0]]
        b = context[self.onnx_node.input[1]]
        context[self.onnx_node.output[0]] = a + b

    def verify_node(self):
        pass
