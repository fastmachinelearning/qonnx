
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
from onnxscript import opset17 as op
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
        

from onnx import helper as oh
import numpy as np
import onnxscript
from onnxscript.ir.passes.common import LiftConstantsToInitializersPass
    
    
def test_copy_metadata_props_batchnorm2affine():
    @script()
    def MyBatchNorm(X: FLOAT[1, 3, 4, 4]) -> FLOAT[1, 3, 4, 4]:
        scale = op.Constant(value=[[1.0, 1.0, 1.0]])
        B = op.Constant(value=[[0.0, 0.0, 0.0]])
        var = op.Constant(value=[[1.0, 1.0, 1.0]])
        mean = op.Constant(value=[[0.0, 0.0, 0.0]])
        return op.BatchNormalization(X, scale, B, mean, var, epsilon=1e-5, momentum=0.9)
    
    # remove cast-like nodes
    model_proto = onnxscript.optimizer.optimize(MyBatchNorm.to_model_proto())    
    
    # batchnorm_to_affine requires initializers for scale/mean/var/bias
    model_ir = onnxscript.ir.serde.deserialize_model(model_proto)
    pass_ = LiftConstantsToInitializersPass(lift_all_constants=True, size_limit=1)
    PassResult = pass_.call(model_ir)
    model_proto = onnxscript.ir.serde.serialize_model(PassResult.model)
    
    # Add metadata to BatchNorm node
    bn_node = model_proto.graph.node[0]
    bn_node.metadata_props.extend([
       add_metadata("key1", "value1"),
       add_metadata("key2", "value2")
    ])
    
    # Create Model Wrapper
    mw = ModelWrapper(model_proto)
    from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
    transformed_mw = mw.transform(BatchNormToAffine())
    
    # Check that metadata was copied
    for node in transformed_mw.graph.node:
        assert node.metadata_props[0].key == 'key1'
        assert node.metadata_props[0].value == 'value1'
        assert node.metadata_props[1].key == 'key2'
        assert node.metadata_props[1].value == 'value2'