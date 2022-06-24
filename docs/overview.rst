*********
Overview
*********

``finn-base`` is part of the `FINN
project <https://xilinx.github.io/finn/>`__ and provides the core
infrastructure for the `FINN
compiler <https://github.com/Xilinx/finn/>`__, including:

-  wrapper around ONNX models for easier manipulation
-  infrastructure for applying transformation and analysis passes on
   ONNX graphs
-  infrastructure for defining and executing custom ONNX ops (for
   verification and code generation)
-  extensions to ONNX models using annotations, including few-bit data
   types, sparsity and data layout specifiers
-  several transformation passes, including topological sorting,
   constant folding and convolution lowering
-  several custom ops including im2col and multi-thresholding for
   quantized activations
-  several utility functions, including packing for few-bit integers

While the FINN compiler itself is focused on producing customized
streaming dataflow hardware architectures from quantized neural network (QNN)
descriptions, the infrastructure offered by ``finn-base`` is more generic and
can be used for developing compilers and code generators for QNN inference.

Intermediate Representation: FINN-ONNX
======================================

We use `ONNX <https://github.com/onnx/onnx>`_ as an intermediate representation (IR) for neural networks. As such, almost every component inside FINN uses ONNX and its `Python API <https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md>`_, so you may want to familiarize yourself with how ONNX represents DNNs. Specifically, the `ONNX protobuf description <https://github.com/onnx/onnx/blob/master/onnx/onnx.proto>`_ (or its `human-readable documentation <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_ and the `operator schemas <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_ are useful as reference documents. We also provide a Jupyter notebook that can help to get familiar with ONNX by showing how to work with a simple ONNX model in FINN, see chapter :ref:`tutorials` for details.

.. note:: FINN uses ONNX is a specific way that we refer to as FINN-ONNX, and not all ONNX graphs are supported by FINN (and vice versa).

Custom Quantization Annotations
===============================

ONNX does not support datatypes smaller than 8-bit integers, whereas in FINN we are interested in smaller integers down to ternary and bipolar. To make this work, FINN uses the quantization_annotation field in ONNX to annotate tensors with their FINN DataType (:py:mod:`finn.core.datatype.DataType`) information. However, all tensors are expected to use single-precision floating point (float32) storage in FINN. This means we store even a 1-bit value as floating point for the purposes of representation. The FINN compiler flow is responsible for eventually producing a packed representation for the target hardware, where the 1-bit is actually stored as 1-bit.

Note that FINN uses floating point tensors as a carrier data type to represent integers. Floating point arithmetic can introduce rounding errors, e.g. (int_num * float_scale) / float_scale is not always equal to int_num.
When using the custom ONNX execution flow, FINN will attempt to sanitize any rounding errors for integer tensors. See (:py:mod:`finn.util.basic.sanitize_quant_values`) for more information.
This behavior can be disabled (not recommended!) by setting the environment variable SANITIZE_QUANT_TENSORS=0.

Custom Operations/Nodes
=======================

FINN uses many custom operations (op_type in ONNX NodeProto) that are not defined in the ONNX operator schema. These custom nodes are marked with domain="finn.*" in the protobuf to identify them as such. These nodes can represent specific operations that we need for low-bit networks, or operations that are specific to a particular hardware backend. To get more familiar with custom operations and how they are created, please take a look in the Jupyter notebook about CustomOps (see chapter :ref:`tutorials` for details) or directly in the module :py:mod:`finn.custom_op`.

.. note:: See the description of `this PR <https://github.com/Xilinx/finn-base/pull/6>`_ for more on how the operator wrapper library is organized.

Custom ONNX Execution Flow
==========================

To verify correct operation of FINN-ONNX graphs, FINN provides its own ONNX execution flow (:py:mod:`finn.core.onnx_exec`). This flow supports the standard set of ONNX operations as well as the custom FINN operations.

.. warning:: This execution flow is only meant for checking the correctness of models after applying transformations, and not for high performance inference.

.. _modelwrapper:

ModelWrapper
============

FINN provides a ModelWrapper class (:py:mod:`finn.core.modelwrapper.ModelWrapper`) as a thin wrapper around ONNX to make it easier to analyze and manipulate ONNX graphs. This wrapper provides many helper functions, while still giving full access to the ONNX protobuf representation.

Some of the helper functions are described in more detail below.

Create a ModelWrapper instance
------------------------------
The ModelWrapper instance can be created using a model in .onnx format or by directly passing a ModelProto instance to the wrapper. The code block below gives an example of how to use the wrapper on a model in .onnx format.
::

  from finn.core.modelwrapper import ModelWrapper
  model = ModelWrapper("model.onnx")

Access the ONNX GraphProto through ModelWrapper
-----------------------------------------------
The ONNX ModelProto can be accessed with following command:
::

  modelproto = model.model

The graph can be accessed using:
::

  graphproto = model.graph

The node list is accessed by:
::

  nodes = model.graph.node

The individual nodes can be selected via their indices.
::

  # first node
  nodes[0]

The number of all nodes can be determined with the len() function in Python.
::

  # number of nodes in the graph
  len(nodes)

Helper functions for tensors
----------------------------

A list of all tensors (names) can easily be accessed using:
::

  tensor_list = model.get_all_tensor_names()

If we take a single tensor from that list (by index), we can determine their producer or consumer node by using one of the following functions. Note that it may be that a tensor does not have a producer or consumer node, for example if the tensor represents a constant that is already set. In that case `None` will be returned.
::

  # find producer of third tensor in model tensor list
  model.find_producer(tensor_list[2])

  # find consumer of third tensor in model tensor list
  model.find_consumer(tensor_list[2])

Every tensor has a specific shape, to get or to set this shape these functions can be used:
::

  # get tensor shape of third tensor in model tensor list
  model.get_tensor_shape(tensor_list[2])

  # set tensor shape of third tensor in model tensor list
  tensor_shape = [1, 1, 28, 28]
  model.set_tensor_shape(tensor_list[2], tensor_shape)

Optionally, the dtype (container datatype) of the tensor can also be specified as third argument in the set function. By default it is set to TensorProto.FLOAT.

As mentioned above there are FINN DataTypes additional to the container datatype, these can be accessed and set for a tensor with the following functions:
::

  # get tensor dataype of third tensor in model tensor list
  model.get_tensor_datatype(tensor_list[2])

  # set tensor datatype of third tensor in model tensor list
  from finn.core.datatype import DataType

  finn_dtype = DataType.BIPOLAR
  model.set_tensor_datatype(tensor_list[2], finn_dtype)

ModelWrapper contains two helper functions for tensor initializers, one to determine the current initializer and one to set the initializer of a tensor. If there is no initializer, None is returned.
::

  # get tensor initializer of third tensor in model tensor list
  model.get_initializer(tensor_list[2])

ModelWrapper contains more useful functions, if you are interested please have a look at the ModelWrapper module (:py:mod:`finn.core.modelwrapper.ModelWrapper`) directly.


.. _analysis_pass:

Analysis Pass
=============

An analysis pass traverses the graph structure and produces information about certain properties. It gets the model in the ModelWrapper as input and returns a dictionary of the properties the analysis extracts. If you are interested in how to write an analysis pass for FINN, please take a look at the Jupyter notebook about how to write an analysis pass, see chapter :ref:`tutorials` for details. For more information about existing analysis passes in finn-base, see module :py:mod:`finn.analysis`.

.. _transformation_pass:

Transformation Pass
===================

A transformation passes changes (transforms) the given model, it gets the model in the ModelWrapper as input and returns the changed model (ModelWrapper) to the FINN flow. Additional the flag *model_was_changed* which indicates if a transformation has to be performed more than once, is returned. If you are interested in how to write a transformation pass for FINN, please take a look at the Jupyter notebook about how to write a transformation pass, see chapter :ref:`tutorials` for details. For more information about existing transformation passes in finn-base, see module :py:mod:`finn.transformation`.
