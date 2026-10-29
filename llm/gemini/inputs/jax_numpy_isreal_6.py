
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import jax.core
import jax.numpy as jnp
from jax.tree_util import register_pytree_node
import numpy as np


class RealTuple(tuple, jax.Array):

    def __new__(cls, iterable=(), dtype=np.float32):
        arr = np.asarray(iterable, dtype=dtype)
        obj = super().__new__(cls, iterable)
        obj._shape = arr.shape
        obj._dtype = arr.dtype
        obj._aval = jax.core.ShapedArray(obj._shape, obj._dtype)
        return obj

    def __deepcopy__(self, memo):
        return RealTuple(copy.deepcopy(tuple(self), memo), dtype=self._dtype)

    def __reduce__(self):
        return (RealTuple, (tuple(self), self._dtype))

    @property
    def aval(self):
        return self._aval

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return int(np.prod(self._shape)) if self._shape else 0

    def min(self, *args, **kwargs):
        return np.asarray(tuple(self)).min(*args, **kwargs)

    def max(self, *args, **kwargs):
        return np.asarray(tuple(self)).max(*args, **kwargs)

    def all(self, *args, **kwargs):
        return np.asarray(tuple(self)).all(*args, **kwargs)

    def any(self, *args, **kwargs):
        return np.asarray(tuple(self)).any(*args, **kwargs)

    def __array__(self, dtype=None):
        return np.asarray(tuple(self), dtype=dtype or self._dtype)


try:
    jax.Array.register(RealTuple)
except Exception:
    pass

try:
    register_pytree_node(
        RealTuple,
        lambda x: (list(x), (x.shape, x.dtype)),
        lambda aux, children: RealTuple(children, dtype=aux[1]),
    )
except Exception:
    pass


def isreal_inputs():
    list_of_inputs = []

    # Input 1: 1D tuple of integers
    input_dict = {"x": RealTuple((1, -2, 3, 0, -10), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tuple of floats
    input_dict = {"x": RealTuple((0.0, -1.5, 3.14, -100.0, 50.5), dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tuple of booleans
    input_dict = {
        "x": RealTuple((True, False, True, True, False), dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: single element tuple
    input_dict = {"x": RealTuple((42,), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D nested tuple of integers
    input_dict = {"x": RealTuple(((1, 2, 3), (4, 5, 6)), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D nested tuple of floats
    input_dict = {"x": RealTuple(((1.0, -2.5), (3.7, 4.2)), dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: empty tuple
    input_dict = {"x": RealTuple((), dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D nested tuple of floats
    input_dict = {
        "x": RealTuple(
            (
                ((-1.0, 2.5), (3.0, -4.2)),
                ((5.1, -6.0), (7.7, 8.8)),
            ),
            dtype=np.float32,
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D nested tuple of positive and negative floats
    input_dict = {
        "x": RealTuple(
            (
                (-10.5, 0.0),
                (25.0, -0.5),
            ),
            dtype=np.float32,
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D tuple with negative integers
    input_dict = {
        "x": RealTuple((-100, -200, -300, -400), dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D nested tuple of integers
    input_dict = {
        "x": RealTuple(
            (
                ((((1, 2),),),),
                ((((3, 4),),),),
            ),
            dtype=np.int32,
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isreal_6"] = isreal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isreal_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isreal_6'.")


check_valid('jax.numpy.isreal', generated_inputs['jax.numpy.isreal_6'], lib="jax", suffix=6)
