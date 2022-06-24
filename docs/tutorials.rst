.. _tutorials:

*********
Tutorials
*********

The FINN compiler repository FINN provides several Jupyter notebooks that can help to get familiar with the basics, the internals and the end-to-end flow in FINN.
All Jupyter notebooks can be found in the FINN compiler repository under the `notebook folder <https://github.com/Xilinx/finn/tree/master/notebooks>`_.
Some of those notebooks are specific to FPGA dataflow-style deployment,
but the ones highlighted below are more generic, relating to the core
infrastructure that ``finn-base`` provides.

Basics
======

The notebooks in this folder should give a basic insight into FINN, how to get started and the basic concepts.

* 0_how_to_work_with_onnx

  * This notebook can help you to learn how to create and manipulate a simple ONNX model, also by using FINN

Advanced
========

The notebooks in this folder are more developer oriented. They should help you to get familiar with the principles in FINN and how to add new content regarding these concepts.

* 0_custom_analysis_pass

  * Explains what an analysis pass is and how to write one for FINN.

* 1_custom_transformation_pass

  * Explains what a transformation pass is and how to write one for FINN.

* 2_custom_op

  * Explains the basics of FINN custom ops and how to define a new one.
