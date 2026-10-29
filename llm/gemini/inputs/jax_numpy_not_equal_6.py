
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def not_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array and boolean True
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 1D integer array with 0s and 1s, and boolean False
    x = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D float32 array and boolean True
    x = np.array([[-1.0, 1.0], [0.0, 2.5]], dtype=np.float32)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D int64 array with negative values and boolean False
    x = np.array([[[-2, 0], [1, -1]], [[0, 3], [-5, 0]]], dtype=np.int64)
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar array and boolean True
    x = np.array(1, dtype=np.int32)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 2D boolean array and boolean False
    x = np.array(
        [[True, True], [False, False], [True, False]], dtype=np.bool_
    )
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 4D float64 array and boolean True
    x = np.zeros((2, 1, 3, 2), dtype=np.float64)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D float array with NaNs and Infs and boolean False
    x = np.array([np.nan, np.inf, -np.inf, 0.0, 1.0], dtype=np.float32)
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Empty 2D array and boolean True
    x = np.empty((0, 5), dtype=np.int32)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 1D uint8 array and boolean False
    x = np.array([0, 255, 1, 128], dtype=np.uint8)
    y = False
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 2D complex64 array and boolean True
    x = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [1.0 + 1.0j, 0.0 + 1.0j]], dtype=np.complex64)
    y = True
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.not_equal_6"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_6'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_6'], lib="jax", suffix=6)
