
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_count_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with positive values
    x = np.array([64, 32, 31, 20], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D int32 array with negative and positive values
    x = np.array([-16, -7, 7, 0], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D int16 array
    x = np.array([[2, -7], [-9, 7]], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D int8 array
    x = np.array([[[1, 2], [3, 4]], [[-5, -6], [-7, -8]]], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D int64 array (scalar as tensor)
    x = np.array(42, dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: Another 1D int32 array with different values
    x = np.array([1024, 2048, 4095, 0], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Large 1D int64 array with positive and negative values
    x = np.array([2**30 - 1, -(2**30 - 1), 0, -1], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D int8 array
    x = np.array([0, 1, 2, 3, 4, 127], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D int32 array generated randomly
    x = np.random.randint(-100, 100, size=(2, 2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D int16 array
    x = np.array([32767, 16384, 0], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: High-dimensional array of int64
    x = np.random.randint(-1000, 1000, size=(1, 5, 5), dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_count"] = bitwise_count_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_count' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_count'.")


check_valid('jax.numpy.bitwise_count', generated_inputs['jax.numpy.bitwise_count'], lib="jax", suffix=0)
