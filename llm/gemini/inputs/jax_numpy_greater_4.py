
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def greater_inputs():
    list_of_inputs = []

    # Input 1: 1D float array with positive float scalar
    x = 0.5
    y = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D integer array with negative float scalar
    x = -2.5
    y = np.array([[-3, -2], [-1, 0]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D scalar array with float
    x = 10.0
    y = np.array(5.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D float array with zero
    x = 0.0
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 1D array with special values (inf, -inf, nan)
    x = 1.0
    y = np.array([np.inf, -np.inf, np.nan, 1.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: Large 4D tensor with float64
    x = 3.14159
    y = np.zeros((2, 2, 3, 3), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Empty 1D tensor
    x = 42.0
    y = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 2D boolean array (can be compared with float)
    x = 0.5
    y = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Large positive float scalar with int64 tensor
    x = 1e9
    y = np.array([[1000, 2000], [3000, 4000]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Negative float scalar with uint8 tensor
    x = -0.01
    y = np.array([0, 1, 2, 255], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Float with 5D tensor
    x = -100.5
    y = np.ones((1, 2, 1, 3, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.greater_4"] = greater_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_4'.")


check_valid('jax.numpy.greater', generated_inputs['jax.numpy.greater_4'], lib="jax", suffix=4)
