from qonnx.custom_op.nhwc.wrapped_ops import BatchNormalization, Conv, MaxPool

custom_op = dict()

custom_op["Conv"] = Conv
custom_op["MaxPool"] = MaxPool
custom_op["BatchNormalization"] = BatchNormalization
