
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def extract_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, size matches condition, some padding
    condition = np.array([True, False, True, False, True], dtype=bool)
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    size = 5
    fill_value = np.array(-1.0, dtype=np.float32)
    input_dict = {"condition": condition, "arr": arr, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int array, size less than total array size
    condition = np.array([False, True, False, True], dtype=bool)
    arr = np.array([10, 20, 30, 40], dtype=np.int32)
    size = 3
    fill_value = np.array(0, dtype=np.int32)
    input_dict = {"condition": condition, "arr": arr, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional condition and arr (which get flattened), size is max allowed
    condition = np.array([[True, False], [False, True]], dtype=bool)
    arr = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    size = 4
    fill_value = np.array(9.9, dtype=np.float32)
    input_dict = {"condition": condition, "arr": arr, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Condition size > arr size (truncated), size matches truncated size
    condition = np.array([True, True, True, True, True], dtype=bool)
    arr = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    size = 3
    fill_value = np.array(-9.0, dtype=np.float64)
    input_dict = {"condition": condition, "arr": arr, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Arr size > condition size (truncated), size matches truncated size
    condition = np.array([True, False], dtype=bool)
    arr = np.array([100, 200, 300, 400], dtype=np.int64)
    size = 2
    fill_value = np.array(-1, dtype=np.int64)
    input_dict = {"condition": condition, "arr": arr, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64 inputs, no elements matching, all padded
    condition = np.array([False, False, False], dtype=bool)
    arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    size = 2
    fill_value = np.array(0.0, dtype=np.float64)
    input_dict = {"condition": condition, "arr": arr, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional with shape mismatch (flattened), size within truncated limit
    condition = np.ones((3, 2), dtype=bool)
    arr = np.arange(12, dtype=np.int32).reshape(4, 3)
    size = 6
    fill_value = np.array(-999, dtype=np.int32)
    input_dict = {"condition": condition, "arr": arr, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Size is smaller than the number of extracted elements (truncates the output)
    condition = np.array([True, True, True, True], dtype=bool)
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    size = 2
    fill_value = np.array(0.0, dtype=np.float32)
    input_dict = {"condition": condition, "arr": arr, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large arrays with uint8, size within limit
    condition = np.array([True, False, True, False] * 25, dtype=bool)
    arr = np.arange(100, dtype=np.uint8)
    size = 60
    fill_value = np.array(255, dtype=np.uint8)
    input_dict = {"condition": condition, "arr": arr, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int16 type, negative values, size within limit
    condition = np.array([True, True, False, False, True], dtype=bool)
    arr = np.array([-10, -20, -30, -40, -50], dtype=np.int16)
    size = 4
    fill_value = np.array(-1, dtype=np.int16)
    input_dict = {"condition": condition, "arr": arr, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.extract"] = extract_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.extract' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.extract'.")


check_valid('jax.numpy.extract', generated_inputs['jax.numpy.extract'], lib="jax", suffix=0)
