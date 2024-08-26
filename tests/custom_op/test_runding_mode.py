import pytest

import numpy as np

from qonnx.custom_op.general.quant import resolve_rounding_mode


@pytest.mark.parametrize(
    "rmode,exp",
    [
        ("ROUND", np.array([6, 2, 2, 1, 1, -1, -1, -2, -2, -6])),
        ("CEIL", np.array([6, 3, 2, 2, 1, -1, -1, -1, -2, -5])),
        ("FLOOR", np.array([5, 2, 1, 1, 1, -1, -2, -2, -3, -6])),
        ("UP", np.array([6, 3, 2, 2, 1, -1, -2, -2, -3, -6])),
        ("DOWN", np.array([5, 2, 1, 1, 1, -1, -1, -1, -2, -5])),
        ("HALF_UP", np.array([6, 3, 2, 1, 1, -1, -1, -2, -3, -6])),
        ("HALF_DOWN", np.array([5, 2, 2, 1, 1, -1, -1, -2, -2, -5])),
    ],
)
def test_rounding_modes(rmode, exp):
    test_array = np.array([5.5, 2.5, 1.6, 1.1, 1.0, -1.0, -1.1, -1.6, -2.5, -5.5])
    rounding_fn = resolve_rounding_mode(rmode)
    assert np.array_equal(rounding_fn(test_array), exp)
