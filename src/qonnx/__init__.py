"""QONNX package initialization."""

import warnings
from importlib import metadata


def _load_custom_op_entry_points():
    """Import modules registered under the ``qonnx_custom_ops`` entry point."""

    try:
        eps = metadata.entry_points()
        if hasattr(eps, "select"):
            eps = eps.select(group="qonnx_custom_ops")
        else:
            eps = eps.get("qonnx_custom_ops", [])
        for ep in eps:
            try:
                ep.load()
            except Exception as e:  # pragma: no cover - import failure warning
                warnings.warn(f"Failed to load custom op entry point {ep.name}: {e}")
    except Exception as e:  # pragma: no cover - metadata failure warning
        warnings.warn(f"Failed to query custom op entry points: {e}")


_load_custom_op_entry_points()

