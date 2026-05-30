
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unique_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int32 array, size matches unique elements
    ar = np.array([1, 2, 2, 3, 3, 3], dtype=np.int32)
    input_dict = {
        'ar': ar,
        'return_index': True,
        'return_inverse': False,
        'return_counts': True,
        'axis': 0,
        'equal_nan': True,
        'size': 3,
        'fill_value': np.array(-1, dtype=np.int32),
        'sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array with negative values, larger size requiring padding
    ar = np.array([-1.0, 0.0, 2.0, -1.0, 3.5], dtype=np.float32)
    input_dict = {
        'ar': ar,
        'return_index': False,
        'return_inverse': True,
        'return_counts': False,
        'axis': 0,
        'equal_nan': True,
        'size': 6,
        'fill_value': np.array(-99.0, dtype=np.float32),
        'sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int64 array with negative axis and smaller size causing truncation
    ar = np.array([10, 20, 10, 30], dtype=np.int64)
    input_dict = {
        'ar': ar,
        'return_index': True,
        'return_inverse': True,
        'return_counts': True,
        'axis': -1,
        'equal_nan': False,
        'size': 2,
        'fill_value': np.array(0, dtype=np.int64),
        'sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float64 array containing NaNs
    ar = np.array([np.nan, 1.0, np.nan, 2.0], dtype=np.float64)
    input_dict = {
        'ar': ar,
        'return_index': False,
        'return_inverse': False,
        'return_counts': True,
        'axis': 0,
        'equal_nan': True,
        'size': 3,
        'fill_value': np.array(-1.0, dtype=np.float64),
        'sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int32 array, unique along axis 0
    ar = np.array([[1, 2], [3, 4], [1, 2], [5, 6]], dtype=np.int32)
    input_dict = {
        'ar': ar,
        'return_index': True,
        'return_inverse': True,
        'return_counts': True,
        'axis': 0,
        'equal_nan': True,
        'size': 3,
        'fill_value': np.array([0, 0], dtype=np.int32),
        'sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32 array, unique along axis 1
    ar = np.array([[1, 1, 2], [3, 3, 4]], dtype=np.int32)
    input_dict = {
        'ar': ar,
        'return_index': True,
        'return_inverse': False,
        'return_counts': True,
        'axis': 1,
        'equal_nan': True,
        'size': 3,
        'fill_value': np.array([0, 0], dtype=np.int32),
        'sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int32 array acting as boolean
    ar = np.array([1, 0, 1, 1], dtype=np.int32)
    input_dict = {
        'ar': ar,
        'return_index': False,
        'return_inverse': False,
        'return_counts': True,
        'axis': 0,
        'equal_nan': True,
        'size': 2,
        'fill_value': np.array(0, dtype=np.int32),
        'sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array, padding test with size > unique elements
    ar = np.array([1.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {
        'ar': ar,
        'return_index': True,
        'return_inverse': True,
        'return_counts': True,
        'axis': 0,
        'equal_nan': True,
        'size': 4,
        'fill_value': np.array(-1.0, dtype=np.float32),
        'sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int32 array with sorted=False
    ar = np.array([-5, -5, 10, 10, 20], dtype=np.int32)
    input_dict = {
        'ar': ar,
        'return_index': True,
        'return_inverse': True,
        'return_counts': True,
        'axis': 0,
        'equal_nan': True,
        'size': 5,
        'fill_value': np.array(-99, dtype=np.int32),
        'sorted': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int64 array
    ar = np.array([100, 200, 100, 300], dtype=np.int64)
    input_dict = {
        'ar': ar,
        'return_index': True,
        'return_inverse': True,
        'return_counts': True,
        'axis': -1,
        'equal_nan': False,
        'size': 4,
        'fill_value': np.array(0, dtype=np.int64),
        'sorted': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unique"] = unique_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique'.")


check_valid('jax.numpy.unique', generated_inputs['jax.numpy.unique'], lib="jax", suffix=0)
