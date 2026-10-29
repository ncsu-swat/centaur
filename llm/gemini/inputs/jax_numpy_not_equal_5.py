
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def not_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with matches and non-matches
    x = 0
    y = np.array([0, 1, 2, 0, -1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D integer array
    x = 5
    y = np.array([[5, 2], [3, 5]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: Negative integer and 1D float array
    x = -3
    y = np.array([-3.0, 0.0, 3.0, -3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: Boolean array compared with integer
    x = 1
    y = np.array([True, False, True], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D integer array matching scalar
    x = 10
    y = np.full((2, 3, 4), 10, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 2D array spanning negative and positive numbers
    x = -1
    y = np.arange(-5, 5, dtype=np.int32).reshape((2, 5))
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 0D scalar array
    x = 42
    y = np.array(42, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 4D float64 array of zeros
    x = 0
    y = np.zeros((1, 2, 2, 3), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Unsigned 8-bit integer array
    x = 255
    y = np.array([0, 128, 255], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Empty 1D array
    x = 7
    y = np.empty((0,), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D array with random values
    x = -100
    y = np.random.randint(-150, 150, size=(3, 3, 3), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.not_equal_5"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_5'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_5'], lib="jax", suffix=5)
