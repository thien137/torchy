# torchy: Mini Pytorch
```bash
torchy/
├── __init__.py                 # Initialize the package
├── _tensor.py                  # Contains the Tensor class
├── _numpy/                     # Contains numpy interface
│   ├── __init__.py
│   ├── _binary_ufuncs_impl.py      # Autograd Function implementation of binary ufuncs 
│   └── _binary_ufuncs.py           # Contains numpy interface for binary ufuncs (e.g., add, subtract, ...)
├── autograd/                   # Contains autograd-related functionality (e.g., gradients, backprop)
│   ├── __init__.py
│   └── function.py                 # Contains Function parent class for custom/builtin functional autograd subclassing
|── nn/                         # Neural network layers and utilities
│   ├── __init__.py
│   └── data.py                     # TODO
├── optim/                      # Contains autograd-related functionality (e.g., gradients, backprop)
│   ├── __init__.py
│   ├── optimizer.py                # Contains the Optimizer abstract parent class
│   └── ...
```

# TODO
- Implement autograd propagation in _propagate_backwward
    - Figure out Gradient Accumulation
- Figure out modules
- Figure out optimizer