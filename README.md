# pybind11_template
Minimal template for a python - C++ - interface using pybind11

Build the python module by running
```con
$ cd build
$ cmake ..
$ make
```

The resulting `.so` file is your python module.

Some vanilla tests/benchmarks comparing matrix multiplications are found in [timing.ipynb](timing.ipynb).