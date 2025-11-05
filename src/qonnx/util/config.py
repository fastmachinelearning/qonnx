# Copyright (c) 2020, Xilinx
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

import json
import onnx

from qonnx.custom_op.registry import getCustomOp

# update this code to handle export configs from subgraphs
# where the subgraph is found in a node's attribute as a graph type
def extract_model_config(model, attr_names_to_extract):
    """Create a dictionary with layer name -> attribute mappings extracted from the
    model. The created dictionary can be later applied on a model with
    qonnx.transform.general.ApplyConfig."""

    cfg = dict()
    cfg["Defaults"] = dict()
    for n in model.graph.node:
        oi = getCustomOp(n)
        layer_dict = dict()
        for attr in n.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:  # Graph type
                # If the attribute is a graph, we need to extract the attributes from the subgraph
                cfg.update(extract_model_config(model.make_subgraph_modelwrapper(attr.g), attr_names_to_extract))
            elif attr.name in attr_names_to_extract:
                # If the attribute name is in the list, we can add it directly
                layer_dict[attr.name] = oi.get_nodeattr(attr.name)
        if len(layer_dict) > 0:
            cfg[n.name] = layer_dict
    return cfg


def extract_model_config_to_json(model, json_filename, attr_names_to_extract):
    """Create a json file with layer name -> attribute mappings extracted from the
    model. The created json file can be later applied on a model with
    qonnx.transform.general.ApplyConfig."""

    with open(json_filename, "w") as f:
        json.dump(extract_model_config(model, attr_names_to_extract), f, indent=2)
