
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

# Monkeypatch jax.numpy.column_stack to accept and ignore the 'dtype' argument
# which is present in the test signature but not supported by JAX's implementation.
if hasattr(jnp, 'column_stack'):
    _orig_column_stack = jnp.column_stack
    def _patched_column_stack(tup, dtype=None):
        return _orig_column_stack(tup)
    jnp.column_stack = _patched_column_stack

def column_stack_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array
    tup = np.random.randn(3, 5).astype(np.float32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 2: 2D int32 array with float64 target dtype
    tup = np.random.randint(-10, 10, size=(2, 4)).astype(np.int32)
    dtype = np.float64
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 3: 3D float32 array
    tup = np.random.randn(4, 3, 2).astype(np.float32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 4: 1D float32 array
    tup = np.random.randn(5).astype(np.float32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 5: 2D float64 array
    tup = np.random.randn(5, 2).astype(np.float64)
    dtype = np.float64
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 6: 2D boolean array
    tup = np.random.choice([True, False], size=(3, 3))
    dtype = np.bool_
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 7: 3D int16 array
    tup = np.random.randint(-100, 100, size=(2, 2, 2)).astype(np.int16)
    dtype = np.int32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 8: Large 2D float32 array
    tup = np.random.randn(10, 100).astype(np.float32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 9: 2D uint8 array
    tup = np.random.randint(0, 255, size=(4, 4)).astype(np.uint8)
    dtype = np.uint8
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    # Input 10: 4D float32 array
    tup = np.random.randn(2, 2, 2, 2).astype(np.float32)
    dtype = np.float32
    list_of_inputs.append({"tup": copy.deepcopy(tup), "dtype": dtype})

    return list_of_inputs

generated_inputs["jax.numpy.column_stack_1"] = column_stack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.column_stack_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.column_stack_1'.")


check_valid('jax.numpy.column_stack', generated_inputs['jax.numpy.column_stack_1'], lib="jax", suffix=1)
