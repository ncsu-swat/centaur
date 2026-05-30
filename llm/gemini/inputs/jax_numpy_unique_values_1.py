
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unique_values_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, size equals unique count, fill_value is scalar array
    x = np.array([3, 1, 2, 1, 3], dtype=np.int32)
    size = 3
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append({"x": x, "size": size, "fill_value": fill_value})

    # Input 2: 1D int32 array, size greater than unique count (padding expected)
    x = np.array([5, 5, 2, 8], dtype=np.int32)
    size = 5
    fill_value = np.array(-99, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "size": size, "fill_value": copy.deepcopy(fill_value)})

    # Input 3: 2D float32 array, size less than unique count (truncation expected)
    x = np.array([[1.5, 2.5], [1.5, 3.5]], dtype=np.float32)
    size = 2
    fill_value = np.array(0.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "size": size, "fill_value": copy.deepcopy(fill_value)})

    # Input 4: 1D float64 array with negative values, size greater than unique count
    x = np.array([-1.2, -3.4, -1.2, -5.6], dtype=np.float64)
    size = 4
    fill_value = np.array(-999.9, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "size": size, "fill_value": copy.deepcopy(fill_value)})

    # Input 5: 3D int64 array, size greater than unique count
    x = np.array([[[1, 2], [3, 4]], [[1, 2], [5, 6]]], dtype=np.int64)
    size = 8
    fill_value = np.array(0, dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "size": size, "fill_value": copy.deepcopy(fill_value)})

    # Input 6: 1D int16 array, size equals unique count
    x = np.array([10, 20, 10, 30, 40], dtype=np.int16)
    size = 4
    fill_value = np.array(0, dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x), "size": size, "fill_value": copy.deepcopy(fill_value)})

    # Input 7: 2D int32 array with many duplicates, small size
    x = np.ones((5, 5), dtype=np.int32)
    size = 1
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "size": size, "fill_value": copy.deepcopy(fill_value)})

    # Input 8: 1D float32 array with duplicates and nan, size greater than unique count
    x = np.array([1.0, 2.0, np.nan, 1.0, np.nan], dtype=np.float32)
    size = 4
    fill_value = np.array(-1.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "size": size, "fill_value": copy.deepcopy(fill_value)})

    # Input 9: Large 1D int32 array, size less than unique count
    x = np.arange(100, dtype=np.int32)
    size = 10
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "size": size, "fill_value": copy.deepcopy(fill_value)})

    # Input 10: 4D float32 array, size greater than unique count
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    size = 3
    fill_value = np.array(9.9, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "size": size, "fill_value": copy.deepcopy(fill_value)})

    return list_of_inputs

generated_inputs["jax.numpy.unique_values_1"] = unique_values_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_values_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_values_1'.")


check_valid('jax.numpy.unique_values', generated_inputs['jax.numpy.unique_values_1'], lib="jax", suffix=1)
