
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def asinh_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive floats
    input_dict = {"x": np.array([1.0, 2.0, 3.5, 4.2], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of negative and positive integers
    input_dict = {"x": np.array([-5, -2, 0, 3, 8], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of small float values
    input_dict = {"x": np.array([1e-5, -2e-5, 3e-4], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of floats
    input_dict = {"x": np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of integers
    input_dict = {"x": np.array([[-10, 20], [30, -40]], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array of floats
    input_dict = {"x": np.array([[[0.1, 0.2], [0.3, 0.4]], [[-0.5, -0.6], [-0.7, -0.8]]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single-element array
    input_dict = {"x": np.array([0.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large float values
    input_dict = {"x": np.array([1000.0, -5000.0, 10000.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with float16
    input_dict = {"x": np.array([1.0, -2.5, 3.0, -4.8], dtype=np.float16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array with single row
    input_dict = {"x": np.array([[-1.5, 2.5, -3.5]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.asinh_4"] = asinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asinh_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asinh_4'.")


check_valid('jax.numpy.asinh', generated_inputs['jax.numpy.asinh_4'], lib="jax", suffix=4)
