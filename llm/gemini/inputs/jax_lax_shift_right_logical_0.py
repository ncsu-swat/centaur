
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def shift_right_logical_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D arrays of int32
    x = np.array([16, 32, 64, 128, 256], dtype=np.int32)
    y = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D arrays of int32
    x = np.array([[1024, 2048], [4096, 8192]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 3D array of int32 with 0D scalar array
    x = np.arange(8, dtype=np.int32).reshape((2, 2, 2))
    y = np.array(2, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: Negative values in int32
    x = np.array([-1, -2, -4, -8], dtype=np.int32)
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: 4D arrays with matching dimensions, int32
    x = np.random.randint(0, 1000, size=(2, 2, 2, 2), dtype=np.int32)
    y = np.random.randint(0, 5, size=(2, 2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: Large 1D array of int64
    x = np.random.randint(-100000, 100000, size=(100,), dtype=np.int64)
    y = np.random.randint(0, 60, size=(100,), dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: int64 matrices
    x = np.random.randint(0, 1000000, size=(5, 5), dtype=np.int64)
    y = np.random.randint(0, 64, size=(5, 5), dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: 3D broadcast compatible arrays, int32
    x = np.random.randint(-100, 100, size=(3, 1, 4), dtype=np.int32)
    y = np.random.randint(0, 15, size=(1, 5, 4), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: Scalar 0D arrays, int32
    x = np.array(4096, dtype=np.int32)
    y = np.array(4, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: 2D broadcasting with matching number of dimensions, int32
    x = np.array([[16, 32, 64]], dtype=np.int32) # (1, 3)
    y = np.array([[1], [2]], dtype=np.int32)     # (2, 1)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.lax.shift_right_logical"] = shift_right_logical_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.shift_right_logical' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.shift_right_logical'.")


check_valid('jax.lax.shift_right_logical', generated_inputs['jax.lax.shift_right_logical'], lib="jax", suffix=0)
