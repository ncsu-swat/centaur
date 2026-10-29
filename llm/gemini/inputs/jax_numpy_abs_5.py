
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def abs_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers with negative values
    x = np.array([-1, -2, 3, -4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array of floats with negative values
    x = np.array([-1.5, 2.5, -3.5, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of integers
    x = np.array([[-1, 2], [-3, -4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array of floats
    x = np.array([[-1.0, -2.0, 3.0], [4.0, -5.0, -6.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D array of alternating large integers
    x = np.array([0, -100, 200, -300, 400, -500], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array of integers
    x = np.array([[[-1, 2], [-3, 4]], [[-5, -6], [7, 8]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Single-element 1D array with negative float
    x = np.array([-42.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D array with varied float values
    x = np.array([100.5, -200.25, 300.125, -400.0625], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D array of shape (3, 2)
    x = np.array([[-10, -20], [30, 40], [-50, 60]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D array with large integers
    x = np.array([-1000000, 2000000, -3000000, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D array of floats
    x = np.array(
        [[[-1.1, 2.2], [-3.3, 4.4]], [[-5.5, -6.6], [7.7, -8.8]]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.abs_5"] = abs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.abs_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.abs_5'.")


check_valid('jax.numpy.abs', generated_inputs['jax.numpy.abs_5'], lib="jax", suffix=5)
