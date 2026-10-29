
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import ctypes
import jax.numpy as jnp

class _MappingProxy(ctypes.Structure):
    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("mapping", ctypes.py_object),
    ]

def _tuple_to_jax(self):
    if len(self) == 0:
        return jnp.array([])
    return jnp.stack(self)

_tuple_dict = _MappingProxy.from_address(id(tuple.__dict__)).mapping
_tuple_dict["__jax_array__"] = _tuple_to_jax
_tuple_dict["ravel"] = lambda self: _tuple_to_jax(self).ravel()
ctypes.pythonapi.PyType_Modified(ctypes.py_object(tuple))

def ediff1d_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tuple of integers
    ary = (1, 2, 4, 7, 11)
    to_end = np.array([16, 22], dtype=np.int32)
    to_begin = np.array([0], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 2: 1D tuple of floats with negative values
    ary = (1.5, -2.5, 3.0, -4.5)
    to_end = np.array([0.0], dtype=np.float32)
    to_begin = np.array([-1.0, 0.5], dtype=np.float32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 3: 2D nested tuple
    ary = ((1, 2, 3), (4, 5, 6))
    to_end = np.array([100], dtype=np.int32)
    to_begin = np.array([-100], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 4: Negative integers
    ary = (-5, -10, -15, -20)
    to_end = np.array([-25, -30], dtype=np.int32)
    to_begin = np.array([0, -1], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 5: Single-element tuple
    ary = (42.0,)
    to_end = np.array([1.0, 2.0], dtype=np.float32)
    to_begin = np.array([-1.0], dtype=np.float32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 6: 2D nested tuple
    ary = ((1, 2), (3, 4))
    to_end = np.array([9, 10], dtype=np.int32)
    to_begin = np.array([0], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 7: Tuple of float fractions
    ary = (0.25, -0.75, 1.5, -2.0, 3.25)
    to_end = np.array([5.0], dtype=np.float32)
    to_begin = np.array([-5.0], dtype=np.float32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 8: Float precision
    ary = (1.2, 3.4, -5.6, 7.8)
    to_end = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    to_begin = np.array([0.1, 0.2], dtype=np.float32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 9: Repeated zeros
    ary = (0, 0, 0, 0, 0)
    to_end = np.array([1, 2], dtype=np.int32)
    to_begin = np.array([0], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    # Input 10: Larger range integers
    ary = (100, -200, 300)
    to_end = np.array([400], dtype=np.int32)
    to_begin = np.array([0], dtype=np.int32)
    list_of_inputs.append({"ary": ary, "to_end": to_end, "to_begin": to_begin})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.numpy.ediff1d_5"] = ediff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ediff1d_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ediff1d_5'.")


check_valid('jax.numpy.ediff1d', generated_inputs['jax.numpy.ediff1d_5'], lib="jax", suffix=5)
