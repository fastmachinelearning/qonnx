# SIRA: Scaled-Integer Range Analysis

*We will incorporate changes from SIRA into QONNX and FINN via pull requests once the code is further cleaned up and modularized.
In the meantime, we provide a list of pointers to where the source code is located.*

## Analysis
The main entry point for the analysis is the `range_analysis` function in [this file](src/qonnx/util/range_analysis.py). Given a (Q)ONNX model, this will perform range analysis and return a dictionary of `{tensor_name : RangeInfo}`. The `RangeInfo` dataclass is defined in the same file and captures the information about the range, integer range, scale, bias and more. The [unit tests](tests/analysis/test_range_analysis.py) can serve as further reference.

## Scale and bias aggregation
The `Streamline` transformation in [this file](src/qonnx/transformation/streamline.py) provides scale-and-bias aggregation based on SIRA. The [unit tests](tests/transformation/test_streamline.py) serve as further reference.

## Threshold conversion
A more comprehensive version of the threshold conversion procedure, including unit tests for various activation functions beyond ReLU, can be found [here](https://github.com/iksnagreb/activations).

## Other functionality in SIRA paper
* All QONNX models used for QNN workloads can be found in the [QONNX model zoo](https://github.com/fastmachinelearning/qonnx_model_zoo).
* The branch of FINN used for integrating SIRA can be found [here](https://github.com/Xilinx/finn/tree/custom/SIRA). A subset of features from SIRA are planned to be upstreamed into mainline FINN, including but not limited to:
    * RTL thresholding kernel [already mainlined](https://github.com/Xilinx/finn/blob/dev/src/finn/custom_op/fpgadataflow/rtl/thresholding_rtl.py).
    * Elementwise meta-kernels [open PR, pending further enhacements](https://github.com/Xilinx/finn/pull/1040).
    * New [frontend flow](https://github.com/Xilinx/finn/blob/custom/SIRA/src/finn/builder/frontend_steps.py) incorporating SIRA and [tests](https://github.com/Xilinx/finn/tree/custom/SIRA/tests/frontend).
