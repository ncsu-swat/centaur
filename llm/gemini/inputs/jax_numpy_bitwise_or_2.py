
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with positive integer
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    y = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int64 array with zero
    x = np.array([[10, 20], [30, 40]], dtype=np.int64)
    y = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 1D int32 array with negative integer
    x = np.array([-5, -1, 0, 4, 15], dtype=np.int32)
    y = -2
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D int32 array with power of two integer
    x = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    y = 16
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar array with integer
    x = np.array(7, dtype=np.int32)
    y = 8
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 2D int32 array with negative integer
    x = np.array([[-128, 0], [64, 127]], dtype=np.int32)
    y = -1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 4D int64 array with large integer
    x = np.ones((2, 2, 2, 2), dtype=np.int64) * 255
    y = 1024
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D int64 array with bit pattern
    x = np.array([5, 10, 15, 20], dtype=np.int64)
    y = 3
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 1D int64 array with hex bit mask
    x = np.array([0x0F0F, 0xF0F0, 0x5555], dtype=np.int64)
    y = 0xAAAA
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D int32 array with large bit flag
    x = np.full((3, 3), 100, dtype=np.int32)
    y = 65535
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_or_2"] = bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_or_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_or_2'.")


check_valid('jax.numpy.bitwise_or', generated_inputs['jax.numpy.bitwise_or_2'], lib="jax", suffix=2)
