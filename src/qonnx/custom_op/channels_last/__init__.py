from qonnx.custom_op.channels_last.wrapped_ops import BatchNormalization, Conv, MaxPool

custom_op = dict()

custom_op["Conv"] = Conv
custom_op["MaxPool"] = MaxPool
custom_op["BatchNormalization"] = BatchNormalization
