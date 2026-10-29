
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_values_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with size larger than unique count
    x = np.array([3, 1, 2, 3, 2, 1], dtype=np.int32)
    input_dict = {"x": x, "size": 5, "fill_value": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D integer array with size equal to unique count
    x = np.array([10, 20, 30, 40], dtype=np.int64)
    input_dict = {"x": x, "size": 4, "fill_value": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array with negative and positive values, size smaller than unique count
    x = np.array([-5, 0, 5, -5, 10, -10], dtype=np.int32)
    input_dict = {"x": x, "size": 3, "fill_value": 999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D integer array
    x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.int32)
    input_dict = {"x": x, "size": 6, "fill_value": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array
    x = np.array(
        [[[1.0, 2.0], [2.0, 3.0]], [[3.0, 4.0], [4.0, 5.0]]], dtype=np.float32
    )
    input_dict = {"x": x, "size": 8, "fill_value": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array where all elements are identical
    x = np.array([7, 7, 7, 7, 7], dtype=np.int32)
    input_dict = {"x": x, "size": 3, "fill_value": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 array with duplicates
    x = np.array([[-1.5, 2.5], [2.5, -1.5]], dtype=np.float64)
    input_dict = {"x": x, "size": 4, "fill_value": -99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D large array of integers
    x = np.random.randint(-100, 100, size=(100,), dtype=np.int32)
    input_dict = {"x": x, "size": 50, "fill_value": -999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D integer array
    x = np.arange(16, dtype=np.int32).reshape(2, 2, 2, 2)
    input_dict = {"x": x, "size": 20, "fill_value": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array
    x = np.array([100.0, 200.0, 300.0, 100.0, 200.0], dtype=np.float32)
    input_dict = {"x": x, "size": 5, "fill_value": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D int64 array with all negative values
    x = np.array([[-10, -20], [-30, -10]], dtype=np.int64)
    input_dict = {"x": x, "size": 5, "fill_value": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.unique_values_2"] = unique_values_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_values_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_values_2'.")


check_valid('jax.numpy.unique_values', generated_inputs['jax.numpy.unique_values_2'], lib="jax", suffix=2)
