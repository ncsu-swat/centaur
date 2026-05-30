
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def square_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive integers
    input_dict = {"x": np.array([1, 2, 3, 4, 5], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of negative integers
    input_dict = {"x": np.array([-1, -5, -10, -100], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of floats
    input_dict = {"x": np.array([0.5, -1.5, 2.25, -3.75], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of complex numbers
    input_dict = {"x": np.array([1+3j, -1j, 2+0j], dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of integers
    input_dict = {"x": np.array([[1, 2], [3, 4]], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array of floats
    input_dict = {"x": np.array([[-0.5, 1.5], [2.5, -3.5]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array of integers
    input_dict = {"x": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array containing zeros
    input_dict = {"x": np.array([0.0, -0.0, 0.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array of booleans
    input_dict = {"x": np.array([True, False, True, True], dtype=np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array of single elements
    input_dict = {"x": np.array([[1], [-2], [3], [-4]], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.square_4"] = square_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.square_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.square_4'.")


check_valid('jax.numpy.square', generated_inputs['jax.numpy.square_4'], lib="jax", suffix=4)
