*********
finn-base
*********

.. note:: **finn-base** is currently under active development. APIs will likely change.

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

Installation
============

Install with full functionality including documentation building:

::

  pip install finn-base[onnx,pyverilator,docs]

Lightweight install for e.g. access to data packing utility functions:

::

  pip install finn-base

Testing
=======

With Docker CE installed, execute the following in the repo root:

::

  ./run-docker.sh tests

Alternatively, pull requests to `dev` will trigger GitHub Actions for the above.


.. toctree::
   :maxdepth: 2
   :hidden:

   Overview <overview>
   Tutorials <tutorials>
   API <api/modules>
   License <license>
   Contributors <authors>
   Index <genindex>


* :ref:`modindex`
* :ref:`search`
