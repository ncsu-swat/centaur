
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def relu6_inputs():
    list_of_inputs = []

    # Input 1: 1D array with negative, within range [0, 6], and above 6 values
    x = np.array([-3.0, -0.5, 0.0, 2.5, 6.0, 7.5, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D array (matrix) of float32
    x = np.random.uniform(-10.0, 10.0, size=(4, 5)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar tensor
    x = np.array(4.2, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar tensor (negative)
    x = np.array(-5.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array of float64
    x = np.random.randn(2, 3, 4).astype(np.float64) * 5.0
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array of float16 (typical image tensor shape NCHW)
    x = np.random.uniform(-2.0, 8.0, size=(2, 3, 8, 8)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array of integers
    x = np.array([-10, -1, 0, 1, 5, 6, 7, 20], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D array of int64
    x = np.random.randint(-10, 15, size=(3, 3), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 5D array of float32
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32) * 4.0
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty array
    x = np.empty((0, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Array with special values (zeros, boundaries, large values)
    x = np.array([0.0, 6.0, -np.inf, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.nn.relu6"] = relu6_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.relu6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.relu6'.")


check_valid('jax.nn.relu6', generated_inputs['jax.nn.relu6'], lib="jax", suffix=0)
