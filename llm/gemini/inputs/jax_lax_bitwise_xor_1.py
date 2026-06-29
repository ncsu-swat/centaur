
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int32 arrays
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 6, 7, 8], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D int32 arrays with negative values
    x = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    y = np.array([[5, -6], [-7, 8]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 3D int32 arrays
    x = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int32)
    y = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: Boolean arrays
    x = np.array([True, False, True, False], dtype=bool)
    y = np.array([False, False, True, True], dtype=bool)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: Scalar-like 0D arrays of int64
    x = np.array(42, dtype=np.int64)
    y = np.array(13, dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: Broadcasting (3, 1) and (1, 4) with int32 (same number of dimensions)
    x = np.random.randint(-100, 100, size=(3, 1), dtype=np.int32)
    y = np.random.randint(-100, 100, size=(1, 4), dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: Broadcasting (5, 5) and (5, 1) with int32 (same number of dimensions)
    x = np.random.randint(-1000, 1000, size=(5, 5), dtype=np.int32)
    y = np.random.randint(-1000, 1000, size=(5, 1), dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: Large 1D int64 array
    x = np.random.randint(-100000, 100000, size=(1000,), dtype=np.int64)
    y = np.random.randint(-100000, 100000, size=(1000,), dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: 4D boolean arrays
    x = np.random.choice([True, False], size=(2, 2, 2, 2))
    y = np.random.choice([True, False], size=(2, 2, 2, 2))
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: Large values with int64
    x = np.array([2**60, -2**60], dtype=np.int64)
    y = np.array([2**59, -2**59], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.lax.bitwise_xor_1"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_xor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_xor_1'.")


check_valid('jax.lax.bitwise_xor', generated_inputs['jax.lax.bitwise_xor_1'], lib="jax", suffix=1)
