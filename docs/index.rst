======
QONNX
======

.. note:: **QONNX** is currently under active development. APIs will likely change.

QONNX (Quantized ONNX) introduces three new custom operators -- `Quant <docs/qonnx-custom-ops/quant_op.md>`_, `BipolarQuant <docs/qonnx-custom-ops/bipolar_quant_op.md>`_ and `Trunc <docs/qonnx-custom-ops/trunc_op.md>`_  -- in order to represent arbitrary-precision uniform quantization in ONNX. This enables:

* Representation of binary, ternary, 3-bit, 4-bit, 6-bit or any other quantization.

* Quantization is an operator itself, and can be applied to any parameter or layer input.

* Flexible choices for scaling factor and zero-point granularity.

* Quantized values are carried using standard `float` datatypes to remain ONNX protobuf-compatible.

This repository contains a set of Python utilities to work with QONNX models, including but not limited to:

* executing QONNX models for (slow) functional verification

* shape inference, constant folding and other basic optimizations

* summarizing the inference cost of a QONNX model in terms of mixed-precision MACs, parameter and activation volume

* Python infrastructure for writing transformations and defining executable, shape-inferencable custom ops

* (experimental) data layout conversion from standard ONNX NCHW to custom QONNX NHWC ops


Quickstart
-----------

Operator definitions
+++++++++++++++++++++

* `Quant <docs/qonnx-custom-ops/quant_op.md>`_ for 2-to-arbitrary-bit quantization, with scaling and zero-point

* `BipolarQuant <docs/qonnx-custom-ops/bipolar_quant_op.md>`_  for 1-bit (bipolar) quantization, with scaling and zero-point

* `Trunc <docs/qonnx-custom-ops/trunc_op.md>`_ for truncating to a specified number of bits, with scaling and zero-point

Installation
+++++++++++++

Install latest release from PyPI:

::

   pip install qonnx


Development
++++++++++++

Install in editable mode in a venv:

::

   git clone https://github.com/fastmachinelearning/qonnx
   cd qonnx
   virtualenv -p python3.8 venv
   source venv/bin/activate
   pip install -e .[testing, docs, notebooks]


Run entire test suite, parallelized across CPU cores:

::

   pytest -n auto --verbose



Run a particular test and fall into pdb if it fails:

::

   pytest --pdb -k "test*extend*partition.py::test*extend*partition[extend_id1-2]"



QONNX also uses GitHub actions to run the full test suite on PRs.

.. toctree::
   :maxdepth: 2
   :hidden:

   ONNX-Based Compiler Infrastructure <overview>
   Tutorials <tutorials>
   API <api/modules>
   License <license>
   Contributors <authors>
   Index <genindex>


* :ref:`modindex`
* :ref:`search`
