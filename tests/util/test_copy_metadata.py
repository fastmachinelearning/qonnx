
import onnx
import pytest
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import copy_metadata_props


def add_metadata(key, value):
    return onnx.StringStringEntryProto(key=key, value=value)


def test_copy_metadata_props():
    
    # Create source node with metadata
    src_node = onnx.NodeProto(
        metadata_props=[add_metadata("key1", "value1"), add_metadata("key2", "value2")]
    )
    dst_node = onnx.NodeProto()
    
    copy_metadata_props(src_node, dst_node)
    
    assert len(dst_node.metadata_props) == 2
    assert dst_node.metadata_props[0].key == "key1"
    assert dst_node.metadata_props[0].value == "value1"
    assert dst_node.metadata_props[1].key == "key2"
    assert dst_node.metadata_props[1].value == "value2"


@pytest.mark.parametrize("mode", ["keep_existing", "overwrite"])
def test_copy_metadata_props_existing_target_md(mode):
    
    # Create source node with metadata
    src_node = onnx.NodeProto(
        metadata_props=[add_metadata("key1", "value1"), add_metadata("key2", "value2")]
    )
    # Create destination node with existing metadata
    dst_node = onnx.NodeProto(
        metadata_props=[add_metadata("key1", "value3")]
    )
    
    copy_metadata_props(src_node, dst_node, mode=mode)
    
    assert len(dst_node.metadata_props) == 2
    assert dst_node.metadata_props[0].key == "key1"
    
    if mode == "keep_existing":
        assert dst_node.metadata_props[0].value == "value3"  # Should keep existing
    elif mode == "overwrite":
        assert dst_node.metadata_props[0].value == "value1"  # Should be overwritten
        
    assert dst_node.metadata_props[1].key == "key2"
    assert dst_node.metadata_props[1].value == "value2"
    
    
def test_copy_metadata_props_bad_mode():
    src_node = onnx.NodeProto(
        metadata_props=[add_metadata("key1", "value1")]
    )
    dst_node = onnx.NodeProto()
    
    with pytest.raises(AssertionError):
        copy_metadata_props(src_node, dst_node, mode="invalid_mode")
        

from onnxscript import script
from onnxscript import opset9 as op
from onnxscript import FLOAT
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
        
def test_copy_metadata_props_gemm2matmul():

    @script()
    def MyGemm(A: FLOAT[4, 5], B: FLOAT[5, 4], C: FLOAT[4, 4]) -> FLOAT[4, 4]:
        return op.Gemm(A, B, C)

    model_proto = MyGemm.to_model_proto()
    gemm_node = model_proto.graph.node[0]
    gemm_node.metadata_props.extend([
       add_metadata("key1", "value1"),
       add_metadata("key2", "value2")
    ])

    # Create Model Wrapper
    mw = ModelWrapper(model_proto)
    
    transformed_mw = mw.transform(GemmToMatMul())
   
    for node in transformed_mw.graph.node:
        assert node.metadata_props[0].key == 'key1'
        assert node.metadata_props[0].value == 'value1'
        assert node.metadata_props[1].key == 'key2'
        assert node.metadata_props[1].value == 'value2'