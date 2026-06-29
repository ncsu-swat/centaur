
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def shift_right_arithmetic_inputs():
    list_of_inputs = []

    # Input 1: Basic int32 1D arrays
    x = np.array([16, 32, 64, 128], dtype=np.int32)
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: Negative values (arithmetic shift should preserve sign)
    x = np.array([-16, -32, -64, -128], dtype=np.int32)
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: int32 2D arrays
    x = np.array([[10, -20, 30], [-40, 50, -60]], dtype=np.int32)
    y = np.array([[1, 2, 1], [2, 1, 2]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: int32 3D arrays
    x = np.array([[[100, -200], [300, -400]], [[500, -600], [700, -800]]], dtype=np.int32)
    y = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: int64 1D arrays
    x = np.array([1000000, -2000000, 3000000], dtype=np.int64)
    y = np.array([10, 20, 5], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: int64 2D arrays
    x = np.array([[1024, -2048], [4096, -8192]], dtype=np.int64)
    y = np.array([[2, 3], [4, 5]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: Broadcasting (1, 4) and (3, 4) in int32
    x = np.array([[16, 32, 64, 128]], dtype=np.int32)
    y = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: Broadcasting (3, 1) and (3, 5) in int64
    x = np.array([[-16], [-32], [-64]], dtype=np.int64)
    y = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: 1-element arrays
    x = np.array([[-1024]], dtype=np.int32)
    y = np.array([[5]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: int64 3D arrays
    x = np.array([[[16, 32, 64]]], dtype=np.int64)
    y = np.array([[[1, 2, 3]]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.lax.shift_right_arithmetic"] = shift_right_arithmetic_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.shift_right_arithmetic' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.shift_right_arithmetic'.")


check_valid('jax.lax.shift_right_arithmetic', generated_inputs['jax.lax.shift_right_arithmetic'], lib="jax", suffix=0)
