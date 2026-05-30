
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arccosh_inputs():
    list_of_inputs = []

    # Input 1: Python standard integer
    input_dict = {"x": int(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: np.int32 scalar
    input_dict = {"x": np.int32(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: np.int64 scalar
    input_dict = {"x": np.int64(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of int32 (positive integers >= 1)
    input_dict = {"x": np.array([1, 2, 5, 10], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of int64
    input_dict = {"x": np.array([[2, 3], [4, 5]], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array containing negative, zero, and positive integers
    input_dict = {"x": np.array([-5, 0, 1, 2], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array of int16
    input_dict = {"x": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D array of int32
    input_dict = {"x": np.array(3, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D array of int64 created via arange
    input_dict = {"x": np.arange(1, 15, dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array of int8
    input_dict = {"x": np.ones((2, 2, 2, 2), dtype=np.int8) * 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.arccosh_3"] = arccosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccosh_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccosh_3'.")


check_valid('jax.numpy.arccosh', generated_inputs['jax.numpy.arccosh_3'], lib="jax", suffix=3)
