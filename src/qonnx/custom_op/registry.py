# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

import importlib

from qonnx.util.basic import get_preferred_onnx_opset


def getCustomOp(node, onnx_opset_version=get_preferred_onnx_opset(), brevitas_exception=True):
    "Return a QONNX CustomOp instance for the given ONNX node, if it exists."
    op_type = node.op_type
    domain = node.domain
    if brevitas_exception:
        # transparently resolve Brevitas domain ops to qonnx ones
        domain = domain.replace("onnx.brevitas", "qonnx.custom_op.general")
    try:
        opset_module = importlib.import_module(domain)
        assert type(opset_module.custom_op) is dict, "custom_op dict not found in Python module %s" % domain
        inst_wrapper = opset_module.custom_op[op_type]
        inst = inst_wrapper(node, onnx_opset_version=onnx_opset_version)
        return inst
    except ModuleNotFoundError:
        raise Exception("Could not load custom opset %s, check your PYTHONPATH" % domain)
    except KeyError:
        raise Exception("Op %s not found in custom opset %s" % (op_type, domain))
