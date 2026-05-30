
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmax_inputs():
    list_of_inputs = []

    # Input 1: positive integer, 1D float array with positive/negative values
    x1 = 5
    x2 = np.array([-1.0, 2.5, 6.0, 4.3, -5.2], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative integer, 2D float array with nan and inf
    x1 = -3
    x2 = np.array([[np.nan, 2.0], [np.inf, -10.0]], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero, 3D float array
    x1 = 0
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: large integer, 1D integer array
    x1 = 100
    x2 = np.array([50, 150, 0, -100, 200], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative integer, 4D float array
    x1 = -10
    x2 = np.random.randn(2, 2, 2, 2).astype(np.float32) * 20.0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: positive integer, 1D float array containing NaN
    x1 = 4
    x2 = np.array([np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: negative integer, 2D float array with negative infinity
    x1 = -100
    x2 = np.array([[-np.inf, -200.0], [5.0, -np.inf]], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: positive integer, 2D float16 array
    x1 = 7
    x2 = np.array([[1.0, 8.0], [7.0, 6.0]], dtype=np.float16)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: negative integer, 3D int32 array
    x1 = -5
    x2 = np.array([[[-10, 0], [1, -5]], [[-2, -8], [10, -1]]], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: zero, 1D float64 array
    x1 = 0
    x2 = np.array([-1.5, np.nan, 2.5, np.inf, -np.inf], dtype=np.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fmax_5"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_5'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_5'], lib="jax", suffix=5)
