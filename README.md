# Neural Fluid Fields

![](media/demo.gif)

This is a small library for doing fluid simulation with neural fields. 
Check out our review paper, [Neural Fields in Visual Computing and Beyond](https://neuralfields.cs.brown.edu/)
if you want to learn more about neural fields!

## Code Organization

`neuralff` contains the bulk of the library. The library can be installed as a usual Python module,
by doing `python setup.py develop` or any equivalent. 

The library contains of several components:

The `neuralff.ops` module contains the core utility functions. In particular, 
`neuralff/ops/fluid_physics_ops.py` contains PDE loss functions, `neuralff/ops/vector_ops.py` contains
differential operators, and `neuralff/ops/fluid_ops.py` contains functions for performing advection.

These functions generally take as input a `neuralff.Field` class, which can be a grid-based vector field,
or a neural field so long as it can sample vectors on continuous coordinates with some mechanism.

## Running the Demo

The demo is located in `app`. These are standalone demos which use the `neuralff` library to do things like
real-time fluid simulation using neural fields. 

The demo runs on `glumpy` and `pycuda`, which can be annoying to install. To install:

```
git clone https://github.com/inducer/pycuda
git submodule update --recursive --init
python configure.py --cuda-root=$CUDA_HOME --cuda-enable-gl
python setup.py develop
pip install pyopengl
pip install glumpy
```

To run the demo, simply run `python3 app/interactive_app.py`.


