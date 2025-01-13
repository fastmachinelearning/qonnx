# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
# * Neither the name of AMD nor the names of its
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

from onnx import helper as oh

from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph


class InsertIdentityOnAllTopLevelIO(Transformation):
    """
    Transformation that inserts an Identity node on all top-level inputs and outputs
    of the ONNX graph. This can be useful before calling transformations that do not
    gracefully handle edge cases where transformed tensors are top-level inputs or outputs.
    """

    def apply(self, model):
        graph = model.graph
        for inp in graph.input:
            model = model.transform(InsertIdentity(inp.name, "consumer"))
        for out in graph.output:
            model = model.transform(InsertIdentity(out.name, "producer"))
        return model, False


class InsertIdentity(Transformation):
    """
    Transformation that inserts an Identity node in the ONNX graph. For edge cases
    where tensor_name is a graph input and producer_or_consumer is 'producer', the
    graph input will be replaced with a new tensor name <old_name>_identity. For the
    edge case where tensor_name is a graph output and producer_or_consumer is 'consumer',
    the graph output will be replaced with a new tensor name <old_name>_identity

    Parameters:
    tensor_name (str): The name of the tensor where the Identity node will be inserted.
    producer_or_consumer (str): Indicates whether the Identity node will be inserted before ('producer')
                                or after ('consumer') the tensor_name.

    """

    def __init__(self, tensor_name, producer_or_consumer):
        super().__init__()
        self.tensor_name = tensor_name
        self.producer_or_consumer = producer_or_consumer

    def insert_node_before(self, model, tensor):
        graph = model.graph
        new_tensor_name = tensor + "_identity"
        # rewire the tensor's producer to the new tensor
        prod = model.find_producer(tensor)
        if prod is not None:
            prod_outlist = list(prod.output)
            prod.output[prod_outlist.index(tensor)] = new_tensor_name
        else:
            # if the tensor is an input tensor (top-level)
            # update the graph's input
            top_inp_names = [inp.name for inp in graph.input]
            graph.input[top_inp_names.index(tensor)].name = new_tensor_name
        # Create a new node
        identity_node = oh.make_node("Identity", [new_tensor_name], [tensor])
        # Insert the new node
        # we do this late in the process to avoid affecting find_producer
        graph.node.append(identity_node)

    def insert_node_after(self, model, tensor):
        graph = model.graph
        new_tensor_name = tensor + "_identity"
        # rewire the tensor's consumers to the new node
        consumers = model.find_consumers(tensor)
        if consumers == []:
            # if the tensor is an output tensor (top-level)
            # find the graph's output and replace it with the new name
            top_out_name = [out.name for out in graph.output]
            graph.output[top_out_name.index(tensor)].name = new_tensor_name
            # TODO what if feeding multiple graph outputs? seems unlikely...
        else:
            for consumer in consumers:
                consumer_inplist = list(consumer.input)
                consumer.input[consumer_inplist.index(tensor)] = new_tensor_name
        # Create a new node
        # we do this late in the process to avoid affecting find_consumers
        identity_node = oh.make_node("Identity", [tensor], [new_tensor_name])
        # Insert the new node
        graph.node.append(identity_node)

    def apply(self, model):
        # Find the tensor in the graph
        tshape = model.get_tensor_shape(self.tensor_name)
        if tshape is None:
            raise ValueError(f"Tensor '{self.tensor_name}' not found in the graph.")
        tensor = self.tensor_name
        # Insert the Identity node before or after the specified tensor
        if self.producer_or_consumer == "producer":
            self.insert_node_before(model, tensor)
        elif self.producer_or_consumer == "consumer":
            self.insert_node_after(model, tensor)
        else:
            raise ValueError("producer_or_consumer must be either 'producer' or 'consumer'.")

        model = model.transform(SortGraph())
        # important to return run_again=False to avoid infinite loop
        return (model, False)
