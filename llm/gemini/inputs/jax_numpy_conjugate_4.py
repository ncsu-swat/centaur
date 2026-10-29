
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def conjugate_inputs():
    list_of_inputs = []

    # Input 1: 1D array of complex64 numbers
    input_dict = {"x": np.array([1 + 2j, 3 - 4j, -5 + 6j], dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of real floats
    input_dict = {"x": np.array([1.0, -2.5, 3.14, -0.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of integers
    input_dict = {"x": np.array([-5, 0, 10, 42], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of complex128 numbers
    input_dict = {"x": np.array([[1 + 1j, -2 - 2j], [3j, -4j]], dtype=np.complex128)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of float64 numbers
    input_dict = {"x": np.array([[0.5, -1.5, 2.0], [-3.0, 4.2, -5.1]], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array of complex64 numbers
    input_dict = {"x": np.array([[[1j, 2], [3, 4j]], [[-1j, -2], [-3, -4j]]], dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single-element array of complex number
    input_dict = {"x": np.array([2 - 3j], dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array of int64
    input_dict = {"x": np.array([-10, 0, 25], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array of booleans
    input_dict = {"x": np.array([[True, False], [False, True]], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with infinity and nan values
    input_dict = {"x": np.array([np.nan + 1j, np.inf - 2j, -np.inf + 0j], dtype=np.complex128)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D array with complex numbers
    input_dict = {"x": np.zeros((2, 2, 2, 2), dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.conjugate_4"] = conjugate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.conjugate_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.conjugate_4'.")


check_valid('jax.numpy.conjugate', generated_inputs['jax.numpy.conjugate_4'], lib="jax", suffix=4)
