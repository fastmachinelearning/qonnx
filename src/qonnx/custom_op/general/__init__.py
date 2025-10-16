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

from qonnx.custom_op.general.bipolar_quant import BipolarQuant
from qonnx.custom_op.general.debugmarker import DebugMarker
from qonnx.custom_op.general.floatquant import FloatQuant
from qonnx.custom_op.general.genericpartition import GenericPartition
from qonnx.custom_op.general.im2col import Im2Col
from qonnx.custom_op.general.intquant import IntQuant
from qonnx.custom_op.general.maxpoolnhwc import MaxPoolNHWC
from qonnx.custom_op.general.multithreshold import MultiThreshold
from qonnx.custom_op.general.quantavgpool2d import QuantAvgPool2d
from qonnx.custom_op.general.trunc import Trunc_v1, Trunc_v2
from qonnx.custom_op.general.xnorpopcount import XnorPopcountMatMul

custom_op = dict()

custom_op["DebugMarker"] = DebugMarker
custom_op["QuantAvgPool2d"] = QuantAvgPool2d
custom_op["MaxPoolNHWC"] = MaxPoolNHWC
custom_op["GenericPartition"] = GenericPartition
custom_op["MultiThreshold"] = MultiThreshold
custom_op["XnorPopcountMatMul"] = XnorPopcountMatMul
custom_op["Im2Col"] = Im2Col
custom_op["IntQuant"] = IntQuant
custom_op["Quant"] = IntQuant
custom_op["Trunc"] = Trunc_v1
custom_op["BipolarQuant"] = BipolarQuant
custom_op["FloatQuant"] = FloatQuant

custom_op["DebugMarker_v1"] = DebugMarker
custom_op["QuantAvgPool2d_v1"] = QuantAvgPool2d
custom_op["MaxPoolNHWC_v1"] = MaxPoolNHWC
custom_op["GenericPartition_v1"] = GenericPartition
custom_op["MultiThreshold_v1"] = MultiThreshold
custom_op["XnorPopcountMatMul_v1"] = XnorPopcountMatMul
custom_op["Im2Col_v1"] = Im2Col
custom_op["IntQuant_v1"] = IntQuant
custom_op["Quant_v1"] = IntQuant
custom_op["Trunc_v1"] = Trunc_v1
custom_op["BipolarQuant_v1"] = BipolarQuant
custom_op["FloatQuant_v1"] = FloatQuant

custom_op["Trunc_v2"] = Trunc_v2
