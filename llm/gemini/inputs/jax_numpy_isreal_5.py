
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isreal_inputs():
    list_of_inputs = []

    # Input 1: 1D array of floats
    input_dict = {"x": np.array([1.0, 2.5, -3.2, 0.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of integers
    input_dict = {"x": np.array([1, -5, 100, 0], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of complex numbers
    input_dict = {"x": np.array([1 + 2j, 3j, 0j, -5 - 2j], dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed-like float/complex array
    input_dict = {"x": np.array([1.0, 0.0 + 2.0j, 3.0, -4.5, 5.0 + 0.0j])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean array
    input_dict = {"x": np.array([True, False, True], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array of floats
    input_dict = {"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array of complex numbers
    input_dict = {
        "x": np.array([[1 + 1j, 2.0], [3j, 4.0 - 2j]], dtype=np.complex128)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array containing special float values (nan, inf)
    input_dict = {
        "x": np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array of integers
    input_dict = {
        "x": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single element array
    input_dict = {"x": np.array([1.234], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Array of huge numbers
    input_dict = {"x": np.array([1e10, -1e20, 1e30], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isreal_5"] = isreal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isreal_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isreal_5'.")


check_valid('jax.numpy.isreal', generated_inputs['jax.numpy.isreal_5'], lib="jax", suffix=5)
