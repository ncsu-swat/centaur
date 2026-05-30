
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp

_original_column_stack = jnp.column_stack
def _patched_column_stack(tup, dtype=None):
    res = _original_column_stack(tup)
    if dtype is not None:
        return res.astype(dtype)
    return res
jnp.column_stack = _patched_column_stack

def column_stack_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    tup = np.array([1, 2, 3], dtype=np.int32)
    dtype = np.int32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 2: 2D float32 array
    tup = np.random.randn(3, 4).astype(np.float32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 3: 1D float64 array with negative values
    tup = np.array([-1.5, 2.0, -3.5], dtype=np.float64)
    dtype = np.float64
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 4: 3D int16 array
    tup = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int16)
    dtype = np.int32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 5: 2D float32 array, target dtype float64
    tup = np.random.randn(2, 5).astype(np.float32)
    dtype = np.float64
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 6: 1D bool array
    tup = np.array([True, False, True], dtype=bool)
    dtype = bool
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 7: 4D float32 array
    tup = np.random.randn(2, 2, 3, 3).astype(np.float32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 8: 2D int64 array
    tup = np.random.randint(-100, 100, size=(5, 2)).astype(np.int64)
    dtype = np.int64
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 9: 2D uint8 array
    tup = np.random.randint(0, 255, size=(4, 4)).astype(np.uint8)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 10: 1D float32 array
    tup = np.array([0.0, -0.0, np.inf], dtype=np.float32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    return list_of_inputs

generated_inputs["jax.numpy.column_stack_2"] = column_stack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.column_stack_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.column_stack_2'.")


check_valid('jax.numpy.column_stack', generated_inputs['jax.numpy.column_stack_2'], lib="jax", suffix=2)
