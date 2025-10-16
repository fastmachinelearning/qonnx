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


def getCustomOp(node, onnx_opset_version=None, brevitas_exception=True):
    "Return a QONNX CustomOp wrapper for the given ONNX node and given opset version,"
    "if it exists. If opset version is None, the default handler for the op type will be used. "
    "If version is specified but the exact version match isn't available, the highest available version "
    "smaller than the requested version will be used."
    op_type = node.op_type
    domain = node.domain
    if brevitas_exception:
        # transparently resolve Brevitas domain ops to qonnx ones
        domain = domain.replace("onnx.brevitas", "qonnx.custom_op.general")
    try:
        opset_module = importlib.import_module(domain)
        assert isinstance(opset_module.custom_op, dict), "custom_op dict not found in Python module %s" % domain
        found_opset_version = None
        if onnx_opset_version is None:
            inst_wrapper = opset_module.custom_op[op_type]
        else:
            op_type_with_version = op_type + "_v" + str(onnx_opset_version)
            if op_type_with_version in opset_module.custom_op:
                # priority: if it exists, load the versioned CustomOp wrapper
                inst_wrapper = opset_module.custom_op[op_type_with_version]
                found_opset_version = onnx_opset_version
            else:
                # when the exact version match is not found
                # version handling: use highest available version smaller than requested version
                available_versions = [
                    int(k.split("_v")[-1]) for k in opset_module.custom_op.keys() if k.startswith(op_type + "_v")
                ]
                suitable_versions = [v for v in available_versions if v <= onnx_opset_version]
                if suitable_versions:
                    highest_version = max(suitable_versions)
                    inst_wrapper = opset_module.custom_op[f"{op_type}_v{highest_version}"]
                    found_opset_version = highest_version
                else:
                    raise Exception(
                        "Op %s version %s not found in custom opset %s" % (op_type, str(onnx_opset_version), domain)
                    )
        inst = inst_wrapper(node, onnx_opset_version=found_opset_version)
        return inst
    except ModuleNotFoundError:
        raise Exception("Could not load custom opset %s, check your PYTHONPATH" % domain)
    except KeyError:
        raise Exception("Op %s version %s not found in custom opset %s" % (op_type, str(onnx_opset_version), domain))
