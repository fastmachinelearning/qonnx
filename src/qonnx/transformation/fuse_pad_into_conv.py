# Copyright (c) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of QONNX nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import onnxoptimizer as opt

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation


class FusePadIntoConv(Transformation):
    def apply(self, model):
        # TODO check if this is fixed in recent versions:
        # the fuse_pad_into_conv from onnxoptimizer does not work
        # properly if the 3rd input of the Pad nodes has an empty string
        # as the tensor name, better to remove it instead
        # (both specify the default padding value as 0)
        for node in model.graph.node:
            if node.op_type == "Pad":
                if len(node.input) == 3 and node.input[2] == "":
                    del node.input[2]
        # the onnxoptimizer pass unfortunately destroys quantization annotations
        # so we make a backup here first
        qnt_annotations = list(model.graph.quantization_annotation)
        model_proto = model.model
        model_proto = opt.optimize(model_proto, passes=["fuse_pad_into_conv"])
        model = ModelWrapper(model_proto)
        model.graph.quantization_annotation.extend(qnt_annotations)
        return (model, False)
