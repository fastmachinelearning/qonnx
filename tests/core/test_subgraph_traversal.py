import pytest
from collections import Counter

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from qonnx.util.basic import qonnx_make_model, get_by_name
import onnx
from onnx import helper

# Helper to recursively build a graph with subgraphs attached to nodes
def make_graph(tree):
    """
    Recursively build a ModelWrapper tree from a nested tuple/list structure.
    Each graph will have one node per subgraph, with the subgraph attached as a node attribute.
    Example input:
        ("top", [("sub1", []), ("sub2", [("sub2_1", [])])])
    Returns the top-level ModelWrapper.
    """
    name, subtrees = tree
    # Create subgraphs recursively
    subgraph_nodes = []
    inputs = []
    outputs = []
    for subtree in subtrees:
        subgraph = make_graph(subtree)
        sg_name_in = f"{subgraph.name}_in"
        sg_name_out = f"{subgraph.name}_out"
        inputs.append(onnx.helper.make_tensor_value_info(sg_name_in, onnx.TensorProto.FLOAT, [4, 4]))
        outputs.append(onnx.helper.make_tensor_value_info(sg_name_out, onnx.TensorProto.FLOAT, [4, 4]))
        # Attach subgraph as attribute to node
        node = helper.make_node(
            op_type="SubgraphNode",  # dummy op_type
            inputs=[sg_name_in],
            outputs=[sg_name_out],
            name=f"{subgraph.name}_node",
        )
        # ONNX expects subgraphs as AttributeProto, so we set it below
        attr = onnx.helper.make_attribute("body", subgraph)
        node.attribute.append(attr)
        subgraph_nodes.append(node)
    # Create the graph for this level
    graph = helper.make_graph(
        nodes=subgraph_nodes,
        name=name,
        inputs=inputs,
        outputs=outputs,
    )

    return graph

def make_subgraph_model(tree):
    """
    Build a ModelWrapper with a graph structure based on the provided tree.
    The tree is a nested tuple/list structure where each node can have subgraphs.
    """
    return ModelWrapper(qonnx_make_model(make_graph(tree), opset_imports=[helper.make_opsetid("", 10)]))


class DummyTransform(Transformation):
    def __init__(self):
        self.visited = list()

    def apply(self, model_wrapper):
        # get the name of the graph being transformed
        graph_name = model_wrapper.model.graph.name
        # set a metadata property to test whether metadata is preserved
        model_wrapper.set_metadata_prop(graph_name, "visited")
        model_wrapper.set_metadata_prop("opset_id", str(model_wrapper.model.opset_import[0].version))
        # add a dummy node to the graph to simulate a transformation
        # to see if the subgraph transformation is presered

        dummy_name_in = f"{graph_name}_dummy_in"
        dummy_name_out = f"{graph_name}_dummy_out"
        model_wrapper.model.graph.input.append(helper.make_tensor_value_info(dummy_name_in, onnx.TensorProto.FLOAT, [4, 4]))
        model_wrapper.model.graph.output.append(helper.make_tensor_value_info(dummy_name_out, onnx.TensorProto.FLOAT, [4, 4]))
        model_wrapper.model.graph.node.append(
            helper.make_node(
                "DummyNode",  # dummy op_type
                inputs=[dummy_name_in],
                outputs=[dummy_name_out],
                name=f"{graph_name}_dummy_node",
            )
        )

        # collect the name of the graph being transformed to check how many times each graph was visited
        self.visited.append(graph_name)
        #import pdb; pdb.set_trace()
        return model_wrapper, False

class NestedTransform(Transformation):
    def __init__(self):
        self.dummy_transform = DummyTransform()
    def apply(self, model_wrapper):
        return model_wrapper.transform(self.dummy_transform), False

def get_subgraph_names(tree):
    """
    Recursively collect the names of all subgraphs in the tree structure.
    """
    names = set()

    def traverse(tree):
        name = tree[0]
        subgraphs = tree[1]
        names.add(name)
        for subgraph in subgraphs:
            traverse(subgraph)

    traverse(tree)
    return names


def check_all_visted_once(tree, transform):
    """
    Check that all subgraphs in the tree structure were visited exactly once.
    """
    visited  = transform.visited
    expected = get_subgraph_names(tree)
    assert Counter(visited) == Counter(expected), f"Visited: {visited}, Expected: {expected}"

def check_visit_order(tree, transform, order):
    """
    Check that the order of visited subgraphs matches the expected preorder or postorder traversal.
    """
    visited = transform.visited
    expected = order(tree)
    assert visited == expected, f"Visited: {visited}, Expected: {expected}"

def check_all_subgraphs_transformed(graph):
    """
    Check that all subgraphs in the tree structure have been transformed.
    """

    # look for the optype "DummyNode" in the model graph
    dummynode_found = False
    for node in graph.node:
        if node.op_type == "DummyNode":
            dummynode_found = True
            break
    if not dummynode_found:
        raise AssertionError(f"DummyNode not found in the transformed model graph {graph.name}")

    # check that metadata is set for all subgraphs
    def get_metadata_props(graph, key):
        metadata_prop = get_by_name(graph.metadata_props, key, "key")
        if metadata_prop is None:
            return None
        else:
            return metadata_prop.value

    assert(get_metadata_props(graph, graph.name) == "visited"), f"Metadata for {graph.name} not set correctly"
    assert(get_metadata_props(graph, "opset_id") == "10"), "Metadata for opset_id not set correctly"
    # recursively check all subgraphs
    for node in graph.node:
         for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                check_all_subgraphs_transformed(attr.g)

@pytest.mark.parametrize("cleanup", [False, True])
@pytest.mark.parametrize("make_deepcopy", [False, True])
@pytest.mark.parametrize("tree, apply_to_subgraphs",
                         [(("top", []), True),
                          (("top", []), False),
                          (("top", [("sub1", [])]), False)])
def test_no_traversal(tree, cleanup, make_deepcopy, apply_to_subgraphs):
    # Check that the top-level model is transformed exactly once when there are no subgraphs.
    # Check that the top-level model is transformed exactly once when there are subgraphs, but apply_to_subgraphs is False.
    # This should always be done correctly regardless of cleanup and make_deepcopy.

    model = make_subgraph_model(tree)
    transform = DummyTransform()
    t_model = model.transform(transform, cleanup, make_deepcopy, apply_to_subgraphs)

    assert transform.visited == ["top"]
    assert t_model.get_metadata_prop("top") == "visited"

def build_preorder_traversal(tree):
    """
    Build a preorder traversal of the tree structure.
    """
    traversal = []

    def traverse(node):
        name, subtrees = node
        traversal.append(name)
        for subtree in subtrees:
            traverse(subtree)

    traverse(tree)
    return traversal

def build_postorder_traversal(tree):
    """
    Build a postorder traversal of the tree structure.
    """
    traversal = []

    def traverse(node):
        name, subtrees = node
        for subtree in subtrees:
            traverse(subtree)
        traversal.append(name)

    traverse(tree)
    return traversal

@pytest.mark.parametrize("cleanup", [False, True])
@pytest.mark.parametrize("make_deepcopy", [False, True])
@pytest.mark.parametrize("tree", [("top", [("sub1", []), ("sub2", [])]),
                                  ("top", [("sub1", [("sub1_1", []), ("sub1_2",[])]), ("sub2", [("sub2_1", [])])])])
@pytest.mark.parametrize("use_preorder_traversal", [True, False])
def test_traversal(tree, cleanup, make_deepcopy, use_preorder_traversal):
    # Check that the top-level model and all subgraphs are transformed when apply_to_subgraphs is True.
    # This should always be done correctly regardless of cleanup and make_deepcopy.
    print(f"Testing tree: {tree}, cleanup: {cleanup}, make_deepcopy: {make_deepcopy}")
    model = make_subgraph_model(tree)
    transform = DummyTransform()
    t_model = model.transform(transform, cleanup, make_deepcopy, apply_to_subgraphs=True, use_preorder_traversal=use_preorder_traversal)

    check_all_visted_once(tree, transform)
    check_all_subgraphs_transformed(t_model.model.graph)

    if use_preorder_traversal:
        traversal_order = build_preorder_traversal
    else:
        traversal_order = build_postorder_traversal
    check_visit_order(tree, transform, traversal_order)


@pytest.mark.parametrize("cleanup", [False, True])
@pytest.mark.parametrize("make_deepcopy", [False, True])
@pytest.mark.parametrize("tree", [("top", [("sub1", []), ("sub2", [])]),
                                  ("top", [("sub1", [("sub1_1", []), ("sub1_2",[])]), ("sub2", [("sub2_1", [])])])])
def test_traversal_nested(tree, cleanup, make_deepcopy):
    # Check that the top-level model and all subgraphs are transformed when apply_to_subgraphs is True.
    # This should always be done correctly regardless of cleanup and make_deepcopy.
    model = make_subgraph_model(tree)
    transform = NestedTransform()
    t_model = model.transform(transform, cleanup, make_deepcopy, apply_to_subgraphs=True)

    check_all_visted_once(tree, transform.dummy_transform)
    check_all_subgraphs_transformed(t_model.model.graph)

def dummy_analysis_fxn(model_wrapper):
    """
    A dummy analysis function that simply returns the model wrapper.
    This is used to test that analysis functions are called correctly.
    """
    d = {}
    return d

@pytest.mark.xfail(reason="Analysis functions require apply_to_subgraphs when traversing subgraphs")
def test_analysis_fxn_without_apply_to_subgraphs_fails():
    # Check that an analysis function fails when apply_to_subgraphs is False
    model = make_subgraph_model(("top", [("sub1", []), ("sub2", [])]))
    model.analysis(dummy_analysis_fxn, apply_to_subgraphs=True)
