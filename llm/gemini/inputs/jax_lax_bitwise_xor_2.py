
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int32 arrays, matching shapes
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([4, 3, 2, 1], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 1D int32 arrays with negative values
    x = np.array([-1, -2, 3, -4], dtype=np.int32)
    y = np.array([4, -3, 2, -1], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 2D int16 arrays
    x = np.array([[12, 15], [24, 30]], dtype=np.int16)
    y = np.array([[15, 12], [30, 24]], dtype=np.int16)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 3D int64 arrays
    x = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int64)
    y = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: Scalar integers
    x = np.array(42, dtype=np.int32)
    y = np.array(13, dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: Broadcasting with same number of dimensions (2D and 2D)
    x = np.random.randint(0, 50, size=(3, 4), dtype=np.int32)
    y = np.random.randint(0, 50, size=(1, 4), dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: Broadcasting 3D array with a 3D single-element array
    x = np.random.randint(-10, 10, size=(2, 2, 2), dtype=np.int32)
    y = np.array([[[5]]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: Unsigned 32-bit integers
    x = np.array([100, 200, 300], dtype=np.uint32)
    y = np.array([300, 200, 100], dtype=np.uint32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Large 64-bit integers
    x = np.array([2**40, 2**45], dtype=np.int64)
    y = np.array([2**40 - 1, 2**45 - 1], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 4D int8 arrays
    x = np.random.randint(-128, 127, size=(2, 2, 2, 2), dtype=np.int8)
    y = np.random.randint(-128, 127, size=(2, 2, 2, 2), dtype=np.int8)
    list_of_inputs.append({"x": x, "y": y})

    # Input 11: Broadcasting matching dimensions (1, 5) and (5, 1)
    x = np.random.randint(0, 10, size=(1, 5), dtype=np.int32)
    y = np.random.randint(0, 10, size=(5, 1), dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs


generated_inputs["jax.lax.bitwise_xor_2"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_xor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_xor_2'.")


check_valid('jax.lax.bitwise_xor', generated_inputs['jax.lax.bitwise_xor_2'], lib="jax", suffix=2)
