
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with positive integer
    x = 1
    y = np.array([0, 1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int64 array with zero
    x = 0
    y = np.array([[1, 2], [3, 4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 1D int32 array with negative integer
    x = -1
    y = np.array([-5, 0, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D int32 array with positive integer
    x = 255
    y = np.array([[[10, 20], [30, 40]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D int64 array with positive integer
    x = 15
    y = np.zeros((2, 3, 4), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 1D int32 array with negative integer
    x = -128
    y = np.array([127, -128, 0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D int32 array with positive integer
    x = 1024
    y = np.arange(12, dtype=np.int32).reshape(3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 0D int32 scalar array
    x = 42
    y = np.array(5, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D int64 array with negative integer
    x = -999
    y = np.ones((2, 2, 2, 2), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D int64 array with large values
    x = 123456
    y = np.array([[-100, 200, -300], [400, -500, 600]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_xor_3"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_xor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_xor_3'.")


check_valid('jax.numpy.bitwise_xor', generated_inputs['jax.numpy.bitwise_xor_3'], lib="jax", suffix=3)
