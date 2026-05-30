
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def flatnonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with mixed zeros and non-zeros
    a = np.array([0, 2, 0, 4, 0, 6], dtype=np.int32)
    input_dict = {"a": a, "size": 3, "fill_value": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array
    a = np.array([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]], dtype=np.float32)
    input_dict = {"a": a, "size": 5, "fill_value": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D boolean array
    a = np.array([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=bool)
    input_dict = {"a": a, "size": 4, "fill_value": -99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with all zeros
    a = np.zeros(10, dtype=np.int32)
    input_dict = {"a": a, "size": 5, "fill_value": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array with no zeros
    a = np.array([1, -2, 3, -4, 5], dtype=np.int32)
    input_dict = {"a": a, "size": 3, "fill_value": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64 array
    a = np.array([[0.0, -0.5], [1.2, 0.0]], dtype=np.float64)
    input_dict = {"a": a, "size": 2, "fill_value": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D int32 array with sparse elements
    a = np.zeros((2, 2, 2, 2), dtype=np.int32)
    a[0, 1, 0, 1] = 5
    a[1, 0, 1, 0] = -3
    input_dict = {"a": a, "size": 4, "fill_value": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large 1D array with specified small size
    a = np.arange(100, dtype=np.int32)
    input_dict = {"a": a, "size": 10, "fill_value": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int32 array
    a = np.array([[10, 20], [0, 30]], dtype=np.int32)
    input_dict = {"a": a, "size": 2, "fill_value": 999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array with small size
    a = np.array([[[0.0], [1.1]], [[2.2], [0.0]]], dtype=np.float32)
    input_dict = {"a": a, "size": 1, "fill_value": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.flatnonzero_2"] = flatnonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.flatnonzero_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.flatnonzero_2'.")


check_valid('jax.numpy.flatnonzero', generated_inputs['jax.numpy.flatnonzero_2'], lib="jax", suffix=2)
