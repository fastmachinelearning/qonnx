# Copies (deep-copies) python objects
import copy

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Base class for all QONNX graph transformations and some basic cleanup
# transformations
from qonnx.transformation.general import (
    Transformation,
    GiveUniqueNodeNames,
    GiveReadableTensorNames,
)
# Cleanup transformations removing identities like multiplication by one or
# addition of zero
from qonnx.transformation.remove import RemoveIdentityOps
# QONNX graph transformations for annotating the graph with datatype and shape
# information
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes


# Composes graph transformations such that each individual transformation as
# well as the whole sequence is applied exhaustively
class ComposedTransformation(Transformation):
    # Initializes the transformation given a list of transformations
    def __init__(self, transformations: list[Transformation]):
        # Initialize the transformation base class
        super().__init__()
        # Register the list of transformations to be applied in apply()
        self.transformations = transformations

    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all transformations to be applied
        for transformation in self.transformations:
            # Start each transformation on a deep copy of the model to mimic the
            # behavior of ModelWrapper.transform()
            model = copy.deepcopy(model)
            # Exhaustively apply the transformation until it no longer modifies
            # the graph
            while True:
                # Apply the transformation once, reporting back whether any node
                # or pattern has been modified
                model, _graph_modified = transformation.apply(model)
                # Keep track whether the graph has been modified at least once
                graph_modified = graph_modified or _graph_modified
                # Break the loop if this transformation did not change anything
                if not _graph_modified:
                    break
            # Apply the cleanup transformations of the ModelWrapper
            model.cleanup()
            # Apply some further cleanup transformations to the model graph
            # removing some clutter and keeping all names readable and ordered
            # at any time
            model = model.transform(RemoveIdentityOps())
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed by at least one transformation so the whole
        # sequence of transformations will be reapplied
        return model, graph_modified
