======
qonnx
======


Frontend and utilities for QONNX

`pip install qonnx`


# Development

Install in editable mode in a venv:

```
git clone https://github.com/fastmachinelearning/qonnx
cd qonnx
virtualenv -p python3.7 venv
source venv/bin/activate
pip install -e .[testing]
```

Run entire test suite, parallelized across CPU cores:
```
pytest -n auto --verbose
```

Run a particular test and fall into pdb if it fails:
```
pytest --pdb -k "test_extend_partition.py::test_extend_partition[extend_id1-2]"
```
