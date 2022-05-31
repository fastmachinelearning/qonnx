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

Run test suite:
```
pytest -n auto --verbose
```
