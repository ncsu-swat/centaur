
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unique_all_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, size=5, fill_value as 0-D array
    x = np.array([3, 4, 1, 3, 1], dtype=np.int32)
    size = 5
    fill_value = np.array(-1, dtype=np.int32)
    input_dict = {"x": x, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array with negative values, size=3, fill_value as 0-D array
    x = np.array([-1.5, 2.3, -1.5, 0.0, 4.2, 2.3], dtype=np.float32)
    size = 3
    fill_value = np.array(99.0, dtype=np.float32)
    input_dict = {"x": x, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array, size=10, fill_value as 0-D array
    x = np.array([[1, 2], [2, 3], [1, 5]], dtype=np.int32)
    size = 10
    fill_value = np.array(0, dtype=np.int32)
    input_dict = {"x": x, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int64 array, size=4, fill_value as 0-D array
    x = np.array([100, 200, 100, 300, 400, 300], dtype=np.int64)
    size = 4
    fill_value = np.array(-999, dtype=np.int64)
    input_dict = {"x": x, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float64 array, size=8, fill_value as 0-D array
    x = np.array([[[1.0, 2.0], [3.0, 1.0]], [[2.0, 5.0], [6.0, 6.0]]], dtype=np.float64)
    size = 8
    fill_value = np.array(-1.0, dtype=np.float64)
    input_dict = {"x": x, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D uint8 array, size=2, fill_value as 0-D array
    x = np.array([10, 20, 10, 30], dtype=np.uint8)
    size = 2
    fill_value = np.array(0, dtype=np.uint8)
    input_dict = {"x": x, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int16 array, size=6, fill_value as 0-D array
    x = np.array([-5, -5, 10, 20, -10], dtype=np.int16)
    size = 6
    fill_value = np.array(-99, dtype=np.int16)
    input_dict = {"x": x, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array, size=5, fill_value as 0-D array
    x = np.array([[1.1, 2.2], [1.1, 3.3]], dtype=np.float32)
    size = 5
    fill_value = np.array(0.0, dtype=np.float32)
    input_dict = {"x": x, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int32 array (all distinct), size=10, fill_value as 0-D array
    x = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    size = 10
    fill_value = np.array(100, dtype=np.int32)
    input_dict = {"x": x, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array with duplicates, size=4, fill_value as 0-D array
    x = np.array([0.5, 0.5, 0.5, 1.5, 1.5], dtype=np.float32)
    size = 4
    fill_value = np.array(-1.0, dtype=np.float32)
    input_dict = {"x": x, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unique_all_1"] = unique_all_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_all_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_all_1'.")


check_valid('jax.numpy.unique_all', generated_inputs['jax.numpy.unique_all_1'], lib="jax", suffix=1)
