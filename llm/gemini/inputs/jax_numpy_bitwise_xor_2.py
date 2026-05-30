
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int32 array and a positive integer
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    y = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 2: 2D int16 array with negative values and a negative integer
    x = np.array([[10, -20], [30, -40]], dtype=np.int16)
    y = -5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 3: 3D int32 array and a positive integer
    x = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    y = 7
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 4: 1D int64 array with larger values and zero
    x = np.array([-1000000, 2000000, -3000000], dtype=np.int64)
    y = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 5: 4D int8 array and integer 1
    x = np.ones((2, 2, 2, 2), dtype=np.int8)
    y = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 6: 0D array (scalar) of int32 and integer 13
    x = np.array(42, dtype=np.int32)
    y = 13
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 7: 2D int32 array and integer 1
    x = np.array([[1, 0], [0, 1]], dtype=np.int32)
    y = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 8: Random 2D int32 array with both positive and negative values and integer -1
    x = np.random.randint(-1000, 1000, size=(5, 5), dtype=np.int32)
    y = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 9: 3D int8 array and integer 127
    x = np.zeros((3, 3, 3), dtype=np.int8)
    y = 127
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 10: 1D int64 array and integer 4095
    x = np.array([128, 256, 512, 1024], dtype=np.int64)
    y = 4095
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_xor_2"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_xor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_xor_2'.")


check_valid('jax.numpy.bitwise_xor', generated_inputs['jax.numpy.bitwise_xor_2'], lib="jax", suffix=2)
