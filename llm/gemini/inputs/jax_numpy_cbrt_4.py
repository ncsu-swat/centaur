
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cbrt_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive integers
    x = np.array([1, 8, 27, 64, 125], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of negative integers
    x = np.array([-1, -8, -27, -64, -125], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of mixed floats including zero
    x = np.array([0.0, -8.0, 27.0, -125.0, 0.001], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of integers (2x3)
    x = np.array([[216, 125, 64], [-27, -8, -1]], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of float64 (3x2)
    x = np.array(
        [[0.125, -0.001], [1000.0, -27000.0], [1e-6, -1e9]], dtype=np.float64
    )
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array of floats (2x2x2)
    x = np.array(
        [[[1.0, -1.0], [8.0, -8.0]], [[27.0, -27.0], [64.0, -64.0]]],
        dtype=np.float32,
    )
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty 1D array
    x = np.array([], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array of floats (2x2)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array of booleans
    x = np.array([True, False, True], dtype=bool)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array of floats
    x = np.array([[[[1.0, -8.0]]]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array of special floating point values
    x = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 1D single-element array
    x = np.array([-343.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.cbrt_4"] = cbrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cbrt_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cbrt_4'.")


check_valid('jax.numpy.cbrt', generated_inputs['jax.numpy.cbrt_4'], lib="jax", suffix=4)
