
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class TensorList(list):

    @property
    def shape(self):
        if len(self) > 0 and hasattr(self[0], "shape"):
            return (len(self), *self[0].shape)
        return (len(self),)

    @property
    def dtype(self):
        if len(self) > 0 and hasattr(self[0], "dtype"):
            return self[0].dtype
        return np.dtype("bool")

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))


class CallableFunc:

    def __init__(self, fn, order=0):
        self.fn = fn
        self.order = order

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __lt__(self, other):
        other_val = other.order if isinstance(other, CallableFunc) else other
        return self.order < other_val

    def __le__(self, other):
        other_val = other.order if isinstance(other, CallableFunc) else other
        return self.order <= other_val

    def __gt__(self, other):
        other_val = other.order if isinstance(other, CallableFunc) else other
        return self.order > other_val

    def __ge__(self, other):
        other_val = other.order if isinstance(other, CallableFunc) else other
        return self.order >= other_val

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class FuncList(list):

    @property
    def shape(self):
        return (len(self),)

    @property
    def dtype(self):
        return np.dtype("O")

    @property
    def ndim(self):
        return 1

    @property
    def size(self):
        return len(self)


def piecewise_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with 2 conditions and 2 callable functions
    x = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.int32)
    condlist = TensorList([x < 0, x >= 0])
    funclist = FuncList(
        [CallableFunc(lambda v: 0 * v, 0), CallableFunc(lambda v: v, 1)]
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "condlist": condlist, "funclist": funclist})
    )

    # Input 2: 1D float32 array with 2 conditions and 3 functions (with default)
    x = np.linspace(-5.0, 5.0, 11, dtype=np.float32)
    condlist = TensorList([x < -2.0, x > 2.0])
    funclist = FuncList(
        [
            CallableFunc(lambda v: v + 1.0, 0),
            CallableFunc(lambda v: v - 1.0, 1),
            CallableFunc(lambda v: v * 0.0, 2),
        ]
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "condlist": condlist, "funclist": funclist})
    )

    # Input 3: 2D float32 array with 2 conditions
    x = np.array([[-3.0, -1.0, 0.0], [1.0, 2.0, 4.0]], dtype=np.float32)
    condlist = TensorList([x <= 0.0, x > 0.0])
    funclist = FuncList(
        [CallableFunc(lambda v: -v, 0), CallableFunc(lambda v: v * v, 1)]
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "condlist": condlist, "funclist": funclist})
    )

    # Input 4: 1D float64 array with 3 conditions
    x = np.array([-2.5, -0.5, 0.0, 0.5, 2.5], dtype=np.float64)
    condlist = TensorList([x < -1.0, (x >= -1.0) & (x <= 1.0), x > 1.0])
    funclist = FuncList(
        [
            CallableFunc(lambda v: v * 2.0, 0),
            CallableFunc(lambda v: v * 0.0, 1),
            CallableFunc(lambda v: v / 2.0, 2),
        ]
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "condlist": condlist, "funclist": funclist})
    )

    # Input 5: 2D int32 array with 1 condition and default function
    x = np.arange(-6, 6, dtype=np.int32).reshape(3, 4)
    condlist = TensorList([x % 2 == 0])
    funclist = FuncList(
        [CallableFunc(lambda v: v // 2, 0), CallableFunc(lambda v: v * 3, 1)]
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "condlist": condlist, "funclist": funclist})
    )

    # Input 6: 1D int64 array with constant numerical scalar values
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int64)
    condlist = TensorList([x < 0, x >= 0])
    funclist = FuncList([int(-1), int(1)])
    list_of_inputs.append(
        copy.deepcopy({"x": x, "condlist": condlist, "funclist": funclist})
    )

    # Input 7: 3D float32 array with 2 conditions and default function
    x = np.arange(-12, 12, dtype=np.float32).reshape(2, 3, 4)
    condlist = TensorList([x < -4.0, x > 4.0])
    funclist = FuncList(
        [
            CallableFunc(lambda v: -v, 0),
            CallableFunc(lambda v: v * 0.5, 1),
            CallableFunc(lambda v: v * 0.0, 2),
        ]
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "condlist": condlist, "funclist": funclist})
    )

    # Input 8: 1D float32 array with 3 conditions and a default scalar
    x = np.array([-3.0, -1.5, -0.2, 0.2, 1.5, 3.0], dtype=np.float32)
    condlist = TensorList(
        [x < -2.0, (x >= -2.0) & (x < 0.0), (x >= 0.0) & (x < 2.0)]
    )
    funclist = FuncList(
        [
            CallableFunc(lambda v: v - 1.0, 0),
            CallableFunc(lambda v: v * 2.0, 1),
            CallableFunc(lambda v: v * 3.0, 2),
            CallableFunc(lambda v: v * 0.0, 3),
        ]
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "condlist": condlist, "funclist": funclist})
    )

    # Input 9: 2D float64 array with linear piecewise functions
    x = np.linspace(-3.0, 3.0, 8, dtype=np.float64).reshape(2, 4)
    condlist = TensorList([x < 0.0, x >= 0.0])
    funclist = FuncList(
        [CallableFunc(lambda v: -v, 0), CallableFunc(lambda v: v, 1)]
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "condlist": condlist, "funclist": funclist})
    )

    # Input 10: 1D int32 array with 1 condition and default branch
    x = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    condlist = TensorList([x > 2])
    funclist = FuncList(
        [CallableFunc(lambda v: v * v, 0), CallableFunc(lambda v: v + 10, 1)]
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "condlist": condlist, "funclist": funclist})
    )

    return list_of_inputs


generated_inputs["jax.numpy.piecewise_1"] = piecewise_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.piecewise_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.piecewise_1'.")


check_valid('jax.numpy.piecewise', generated_inputs['jax.numpy.piecewise_1'], lib="jax", suffix=1)
