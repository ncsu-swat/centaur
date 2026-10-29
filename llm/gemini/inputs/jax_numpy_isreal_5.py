
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def isreal_inputs():
    list_of_inputs = []

    # Input 1: 1D array of mixed real and complex numbers
    input_dict = {"x": np.array([False, 0j, 1, 2.1, 1 + 2j])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of integers
    input_dict = {"x": np.array([1, -2, 3, -4, 0], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of floats including negative numbers
    input_dict = {"x": np.array([-3.14, 0.0, 2.718, -1.0, 42.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of pure imaginary and complex numbers
    input_dict = {"x": np.array([1j, -2j, 3 + 4j, 5 - 6j], dtype=np.complex128)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array of booleans
    input_dict = {"x": np.array([True, False, True, True, False], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array of floats
    input_dict = {"x": np.array([[1.5, -2.5, 3.0], [4.2, -5.1, 0.0]], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array of complex numbers
    input_dict = {"x": np.array([[1.0 + 0j, 2.0 + 1j], [3.0 - 2j, 4.0 + 0j]], dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array of integers
    input_dict = {"x": np.array([[[1, -2], [3, 4]], [[-5, 6], [7, -8]]], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array containing special float values (nan, inf)
    input_dict = {"x": np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty 1D array
    input_dict = {"x": np.array([], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D array of complex numbers
    input_dict = {"x": np.array([[[1j, 2 + 0j], [3 - 1j, 0j]], [[5.0, 6j], [7 + 2j, 8 - 0j]]], dtype=np.complex128)}
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
