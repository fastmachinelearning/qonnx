import pytest

import numpy as np
import onnx
import onnx.parser as oprs

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model


def generate_random_input(model):
    """
    Creates input dictionary with a random numpy array
    that matches the input tensor shape.
    """
    input_dict = {}
    for i in range(len(model.graph.input)):
        input_node = model.graph.input[i]
        input_node_name = input_node.name
        input_node_shape = model.get_tensor_shape(input_node_name)
        i_val = gen_finn_dt_tensor(DataType["FLOAT32"], input_node_shape)
        input_dict[input_node_name] = i_val
    return input_dict


def set_all_initializers(model):
    """Sets all initializers of the graph to a random value."""
    for n in model.graph.node:
        if len(n.input) > 1 and n.name != "TopK1":
            init_name = n.input[1]
            init_shape = model.get_tensor_shape(init_name)
            init_val = gen_finn_dt_tensor(DataType["FLOAT32"], init_shape)
            model.set_initializer(init_name, init_val)


def create_arbitrary_model(invalid=False):
    """
    Creates arbitrary model for testing the 3D to 4D transform.
    This model is based on a subpart of QuartzNet.
    """

    Mul1_node = onnx.helper.make_node(
        "Mul",
        inputs=["in1_mul1", "in2_mul1"],  # inputs
        outputs=["out1_mul1"],  # outputs
        name="Mul1",  # name
    )

    Conv1_node = onnx.helper.make_node(
        "Conv",
        inputs=["out1_mul1", "in2_conv1"],
        outputs=["out1_conv1"],
        name="Conv1",
        dilations=[1],
        group=1,
        kernel_shape=[1],
        pads=[0, 0],
        strides=[1],
    )

    if invalid is True:  # To make the graph invalid, a ReLU node is added after the Conv node
        Relu1_node = onnx.helper.make_node("Relu", inputs=["out1_conv1"], outputs=["out1_relu1"], name="Relu1")
        Add1_node = onnx.helper.make_node("Add", inputs=["out1_relu1", "in2_add1"], outputs=["out1_add1"], name="Add1")
    else:
        Add1_node = onnx.helper.make_node("Add", inputs=["out1_conv1", "in2_add1"], outputs=["out1_add1"], name="Add1")

    Mul2_node = onnx.helper.make_node("Mul", inputs=["out1_add1", "in2_mul2"], outputs=["out1_mul2"], name="Mul2")

    Transpose1_node = onnx.helper.make_node(
        "Transpose",
        inputs=["out1_mul2"],
        outputs=["out1_transpose1"],
        name="Transpose1",
        perm=[0, 2, 1],
    )

    LogSoftmax1_node = onnx.helper.make_node(
        "LogSoftmax",
        inputs=["out1_transpose1"],
        outputs=["out1_logsoftmax1"],
        name="LogSoftmax1",
        axis=2,
    )

    ArgMax1_node = onnx.helper.make_node(
        "ArgMax",
        inputs=["out1_logsoftmax1"],
        outputs=["out1_argmax1"],
        name="ArgMax1",
        axis=-1,
        keepdims=0,
    )

    # Inputs and outputs
    in1_mul1 = onnx.helper.make_tensor_value_info("in1_mul1", onnx.TensorProto.FLOAT, [1, 1024, 128])
    out1_argmax1 = onnx.helper.make_tensor_value_info("out1_argmax1", onnx.TensorProto.INT64, [1, 128])

    # Value infos
    out1_mul1 = onnx.helper.make_tensor_value_info("out1_mul1", onnx.TensorProto.FLOAT, [1, 1024, 128])
    out1_conv1 = onnx.helper.make_tensor_value_info("out1_conv1", onnx.TensorProto.FLOAT, [1, 29, 128])

    if invalid is True:
        out1_relu1 = onnx.helper.make_tensor_value_info("out1_relu1", onnx.TensorProto.FLOAT, [1, 29, 128])

    out1_add1 = onnx.helper.make_tensor_value_info("out1_add1", onnx.TensorProto.FLOAT, [1, 29, 128])

    out1_mul2 = onnx.helper.make_tensor_value_info("out1_mul2", onnx.TensorProto.FLOAT, [1, 29, 128])
    out1_transpose1 = onnx.helper.make_tensor_value_info("out1_transpose1", onnx.TensorProto.FLOAT, [1, 128, 29])
    out1_logsoftmax1 = onnx.helper.make_tensor_value_info("out1_logsoftmax1", onnx.TensorProto.FLOAT, [1, 128, 29])

    # Initializers
    in2_mul1 = onnx.helper.make_tensor_value_info("in2_mul1", onnx.TensorProto.FLOAT, [1])
    in2_conv1 = onnx.helper.make_tensor_value_info("in2_conv1", onnx.TensorProto.FLOAT, [29, 1024, 1])
    in2_add1 = onnx.helper.make_tensor_value_info("in2_add1", onnx.TensorProto.FLOAT, [1, 29, 1])
    in2_mul2 = onnx.helper.make_tensor_value_info("in2_mul2", onnx.TensorProto.FLOAT, [1])

    list_of_nodes = [
        Mul1_node,
        Conv1_node,
        Add1_node,
        Mul2_node,
        Transpose1_node,
        LogSoftmax1_node,
        ArgMax1_node,
    ]
    list_of_value_infos = [
        out1_mul1,
        out1_conv1,
        out1_add1,
        out1_mul2,
        out1_transpose1,
        out1_logsoftmax1,
        in2_mul1,
        in2_conv1,
        in2_add1,
        in2_mul2,
    ]

    if invalid is True:
        list_of_nodes.insert(2, Relu1_node)
        list_of_value_infos.append(out1_relu1)

    graph = onnx.helper.make_graph(
        nodes=list_of_nodes,
        name="4d_conversion_test_graph",
        inputs=[in1_mul1],
        outputs=[out1_argmax1],
        value_info=list_of_value_infos,
    )
    onnx_model = qonnx_make_model(graph, producer_name="4d_conversion_test-model")
    model = ModelWrapper(onnx_model)
    set_all_initializers(model)

    return model


def create_arbitrary_model_vgg():
    """
    Creates arbitrary model for testing the 3D to 4D transform.
    This model is based on a subpart of VGG10.
    """
    Conv1_node = onnx.helper.make_node(
        "Conv",
        inputs=["in1_conv1", "in2_conv1"],
        outputs=["out1_conv1"],
        name="Conv1",
        dilations=[1],
        group=1,
        kernel_shape=[3],
        pads=[1, 1],
        strides=[1],
    )

    Div1_node = onnx.helper.make_node("Div", inputs=["out1_conv1", "in2_div1"], outputs=["out1_div1"], name="Div1")

    MaxPool1_node = onnx.helper.make_node(
        "MaxPool",
        inputs=["out1_div1"],
        outputs=["out1_maxpool1"],
        name="MaxPool1",
        kernel_shape=[2],
        pads=[0, 0],
        strides=[2],
    )

    Flatten1_node = onnx.helper.make_node(
        "Flatten",
        inputs=["out1_maxpool1"],
        outputs=["out1_flatten1"],
        name="Flatten1",
        axis=1,
    )

    MatMul1_node = onnx.helper.make_node(
        "MatMul",
        inputs=["out1_flatten1", "in2_matmul1"],
        outputs=["out1_matmul1"],
        name="MatMul1",
    )

    TopK1_node = onnx.helper.make_node(
        "TopK",
        inputs=["out1_matmul1", "in2topk1"],
        outputs=["out1_topk1", "out2_topk1"],
        name="TopK1",
        axis=-1,
        largest=1,
        sorted=1,
    )

    # Inputs and outputs
    in1_conv1 = onnx.helper.make_tensor_value_info("in1_conv1", onnx.TensorProto.FLOAT, [1, 64, 16])
    out2_topk1 = onnx.helper.make_tensor_value_info("out2_topk1", onnx.TensorProto.INT64, [1, 3])

    # Value infos
    out1_conv1 = onnx.helper.make_tensor_value_info("out1_conv1", onnx.TensorProto.FLOAT, [1, 64, 16])
    out1_div1 = onnx.helper.make_tensor_value_info("out1_div1", onnx.TensorProto.FLOAT, [1, 64, 16])
    out1_maxpool1 = onnx.helper.make_tensor_value_info("out1_maxpool1", onnx.TensorProto.FLOAT, [1, 64, 8])
    out1_flatten1 = onnx.helper.make_tensor_value_info("out1_flatten1", onnx.TensorProto.FLOAT, [1, 512])
    out1_matmul1 = onnx.helper.make_tensor_value_info("out1_matmul1", onnx.TensorProto.FLOAT, [1, 24])
    out1_topk1 = onnx.helper.make_tensor_value_info("out1_topk1", onnx.TensorProto.FLOAT, [1, 3])

    # Initializers
    in2_conv1 = onnx.helper.make_tensor_value_info("in2_conv1", onnx.TensorProto.FLOAT, [64, 64, 3])
    in2_div1 = onnx.helper.make_tensor_value_info("in2_div1", onnx.TensorProto.FLOAT, [1])
    in2_matmul1 = onnx.helper.make_tensor_value_info("in2_matmul1", onnx.TensorProto.FLOAT, [512, 24])
    in2topk1 = onnx.helper.make_tensor_value_info("in2topk1", onnx.TensorProto.FLOAT, [1])

    list_of_nodes = [
        Conv1_node,
        Div1_node,
        MaxPool1_node,
        Flatten1_node,
        MatMul1_node,
        TopK1_node,
    ]
    list_of_value_infos = [
        out1_conv1,
        out1_div1,
        out1_maxpool1,
        out1_flatten1,
        out1_matmul1,
        out1_topk1,
        in2_conv1,
        in2_div1,
        in2_matmul1,
        in2topk1,
    ]

    graph = onnx.helper.make_graph(
        nodes=list_of_nodes,
        name="4d_conversion_test_graph",
        inputs=[in1_conv1],
        outputs=[out2_topk1],
        value_info=list_of_value_infos,
    )

    opset_imports = [onnx.helper.make_opsetid("", 11)]
    onnx_model = qonnx_make_model(graph, producer_name="4d_conversion_test-model", opset_imports=opset_imports)
    model = ModelWrapper(onnx_model)

    # Fixed TopK initializer (K=3)
    model.set_initializer("in2topk1", np.array([3]))
    set_all_initializers(model)

    return model


def create_conv_upsample():
    input = """
    <
        ir_version: 7,
        opset_import: ["" : 9]
    >
    agraph (float[1,4,256] in0) => (float[1,2,256] out0)
    <
        float[4,1,16] param_c0_weight,
        float[4] param_c0_bias,
        float[2,4,1] param_c1_weight,
        float[2] param_c1_bias,
        float[3] upsample_scales = {1.0,1.0,2.0}
    >
    {
        c0_out = Conv<dilations=[1],group=4,kernel_shape=[16],pads=[7,7],strides=[2]>(in0, param_c0_weight, param_c0_bias)
        c1_out = Conv<dilations=[1],group=1,kernel_shape=[1],pads=[0,0],strides=[1]>(c0_out, param_c1_weight, param_c1_bias)
        out0 = Upsample<mode="nearest">(c1_out, upsample_scales)
    }
    """
    model = oprs.parse_model(input)
    model = ModelWrapper(model)
    model = model.transform(InferShapes())
    for tensor_name in model.get_all_tensor_names():
        if tensor_name.startswith("param_"):
            init_shape = model.get_tensor_shape(tensor_name)
            model.set_initializer(tensor_name, gen_finn_dt_tensor(DataType["FLOAT32"], init_shape))
    return model


@pytest.mark.parametrize("test_model", ["Quartz", "VGG", "ConvUpsample"])
def test_4d_conversion(test_model):
    """
    Test for the 3D to 4D transformation with a valid graph.
    """

    if test_model == "Quartz":
        model = create_arbitrary_model(invalid=False)
    elif test_model == "VGG":
        model = create_arbitrary_model_vgg()
    elif test_model == "ConvUpsample":
        model = create_conv_upsample()
    else:
        raise Exception("Unknown test_model in test_4d_conversion")

    # Inputs
    input_dict = generate_random_input(model)

    # Comparing the outputs of the model before and after the transform
    output_node_name = model.graph.output[0].name
    output_dict = oxe.execute_onnx(model, input_dict, return_full_exec_context=True)
    expected = output_dict[output_node_name]

    model = model.transform(Change3DTo4DTensors())

    for k, v in input_dict.items():
        old_in_name = k
        old_shape = np.shape(v)
        new_in_name = model.graph.input[0].name
        new_shape = old_shape + (1,)
    new_in_val = np.reshape(v, new_shape)
    del input_dict[old_in_name]
    input_dict[new_in_name] = new_in_val

    output_node_name = model.graph.output[0].name
    output_dict = oxe.execute_onnx(model, input_dict, return_full_exec_context=True)
    expected_modified = output_dict[output_node_name]

    expected_modified = np.reshape(expected_modified, np.shape(expected))

    assert (expected == expected_modified).all()


def test_4d_conversion_invalid_nodes():
    """
    Test for the 3D to 4D transformation with an invalid graph.
    """
    model = create_arbitrary_model(invalid=True)

    # Inputs
    input_dict = generate_random_input(model)

    # Initializers
    set_all_initializers(model)

    # Comparing the outputs of the model before and after the transform
    output_node_name = model.graph.output[0].name
    output_dict = oxe.execute_onnx(model, input_dict, return_full_exec_context=True)
    expected = output_dict[output_node_name]

    model = model.transform(Change3DTo4DTensors())

    output_node_name = model.graph.output[0].name
    output_dict = oxe.execute_onnx(model, input_dict, return_full_exec_context=True)
    expected_modified = output_dict[output_node_name]

    expected_modified = np.reshape(expected_modified, np.shape(expected))

    assert (expected == expected_modified).all()
