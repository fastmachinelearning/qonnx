########################################################################
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#########################################################################

import onnx
import numpy as np

class helper :

    def __init__(self) -> None:
        pass

    def create_initializer_tensor(name: str, tensor_array: np.ndarray, data_type: onnx.TensorProto = onnx.TensorProto.FLOAT) -> onnx.TensorProto:
        initializer_tensor = onnx.helper.make_tensor(name=name,
                                                    data_type=data_type,
                                                    dims=tensor_array.shape,
                                                    vals=tensor_array.flatten().tolist())
        return initializer_tensor

    # to check node.i() exists pass tesor_idx=0, node_idx=0
    # to check node.inputs[1].inputs[0] exists pass tesor_idx=1, node_idx=0
    def is_parent_exist(node, tesor_idx, node_idx):
        if len(node.inputs)>tesor_idx and len(node.inputs[tesor_idx].inputs)>node_idx:
            return True
        return False

    def is_child_present(node,tesor_idx, node_idx):
        if len(node.outputs)>tesor_idx and len(node.outputs[tesor_idx].outputs)>node_idx:
            return True
        return False

    def is_attr_exist(node, attr_name):
        try:
            node.attrs[attr_name]
            return True
        except:
            return False
    
    def is_constant_tensor(tensor):
        try:
            tensor.values
            return True
        except:
            return False