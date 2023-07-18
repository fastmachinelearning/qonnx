# Copyright (c) 2023, Advanced Micro Devices, Inc.
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

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation


class ExposeIntermediateTensorsLambda(Transformation):
    def __init__(self, tensor_filter=lambda tname, model: True):
        super().__init__()
        self.tensor_filter = tensor_filter

    def apply(self, model: ModelWrapper):
        all_tensor_names = model.get_all_tensor_names()
        for tname in all_tensor_names:
            if self.tensor_filter(tname, model):
                # check whether this tensor is already in the outputs
                if tname in [x.name for x in model.graph.output]:
                    # already part of outputs, skip
                    continue
                else:
                    # append ValueInfo to outputs
                    tensor_vi = model.get_tensor_valueinfo(tname)
                    model.graph.output.append(tensor_vi)
                    # remove existing ValueInfo to avoid duplicate
                    model.graph.value_info.remove(tensor_vi)

        return (model, False)


class ExposeIntermediateTensorsPatternList(ExposeIntermediateTensorsLambda):
    def pattern_filter(self, tname, model):
        if self.dynamic_only:
            return any([(pat in tname) and (model.get_initializer(tname) is None) for pat in self.pattern_list])
        else:
            return any([(pat in tname) for pat in self.pattern_list])

    def __init__(self, pattern_list, dynamic_only=True):
        self.pattern_list = pattern_list
        self.dynamic_only = dynamic_only
        super().__init__(tensor_filter=self.pattern_filter)
