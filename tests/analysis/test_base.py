"""Test that the AnalysisPass class and its interface with ModelWrapper work correctly."""
from pkgutil import get_data

from qonnx.analysis.base import AnalysisPass
from qonnx.core.modelwrapper import ModelWrapper


class NumberOfNodes(AnalysisPass[int]):
    def __init__(self) -> None:
        super().__init__()

    def analyze(self, model: ModelWrapper, apply_to_subgraphs: bool) -> int:
        return len(model.graph.node)


def test_analysis_pass() -> None:
    data = get_data("qonnx.data", "onnx/mnist-conv/model.onnx")
    assert data is not None
    model = ModelWrapper(data)
    result = model.analysis(NumberOfNodes())
    assert type(result) is int
    assert result != 0
