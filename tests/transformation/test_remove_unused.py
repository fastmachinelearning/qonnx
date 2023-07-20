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
# * Neither the name of qonnx nor the names of its
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


from qonnx.transformation.expose_intermediate import ExposeIntermediateTensorsPatternList
from qonnx.transformation.remove import RemoveUnusedNodes
from qonnx.util.test import download_model


def test_remove_unused_nodes():
    model = download_model("FINN-TFC_W2A2", do_cleanup=True, return_modelwrapper=True)
    orig_output = model.graph.output[0]
    # break out intermediate output
    pattern_list = ["MatMul_0_out0"]
    model = model.transform(ExposeIntermediateTensorsPatternList(pattern_list, dynamic_only=True))
    # remove original top-level output
    model.graph.output.remove(orig_output)
    assert len(model.graph.output) == 1
    assert model.graph.output[0].name == "MatMul_0_out0"
    # call transform to remove the now-dangling tail nodes
    model = model.transform(RemoveUnusedNodes())
    assert len(model.graph.node) == 6
