
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def setdiff1d_inputs():
    list_of_inputs = []

    # Input 1: Basic integer case, size matches output length
    input_dict = {
        "ar1": np.array([1, 2, 3, 4], dtype=np.int32),
        "ar2": np.array([3, 4, 5, 6], dtype=np.int32),
        "assume_unique": False,
        "size": 2,
        "fill_value": np.array(-1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Unique elements assumption with int64, padding required
    input_dict = {
        "ar1": np.array([10, 20, 30], dtype=np.int64),
        "ar2": np.array([20], dtype=np.int64),
        "assume_unique": True,
        "size": 4,
        "fill_value": np.array(-99, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float32 inputs with truncation
    input_dict = {
        "ar1": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        "ar2": np.array([2.5, 4.5], dtype=np.float32),
        "assume_unique": False,
        "size": 1,
        "fill_value": np.array(0.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative integer values
    input_dict = {
        "ar1": np.array([-1, -2, -3, -4], dtype=np.int32),
        "ar2": np.array([-3, -4, -5], dtype=np.int32),
        "assume_unique": True,
        "size": 3,
        "fill_value": np.array(99, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional array (will be flattened)
    input_dict = {
        "ar1": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "ar2": np.array([2, 3], dtype=np.int32),
        "assume_unique": False,
        "size": 5,
        "fill_value": np.array(-1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array with duplicates and float64 type
    input_dict = {
        "ar1": np.array([1.1, 1.1, 2.2, 2.2, 3.3], dtype=np.float64),
        "ar2": np.array([1.1, 4.4], dtype=np.float64),
        "assume_unique": False,
        "size": 3,
        "fill_value": np.array(-1.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty first array
    input_dict = {
        "ar1": np.array([], dtype=np.int32),
        "ar2": np.array([1, 2], dtype=np.int32),
        "assume_unique": False,
        "size": 2,
        "fill_value": np.array(-1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty second array
    input_dict = {
        "ar1": np.array([5, 6, 7], dtype=np.int32),
        "ar2": np.array([], dtype=np.int32),
        "assume_unique": True,
        "size": 3,
        "fill_value": np.array(-1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complete overlap of elements
    input_dict = {
        "ar1": np.array([100, 200, 300], dtype=np.int32),
        "ar2": np.array([100, 200, 300], dtype=np.int32),
        "assume_unique": False,
        "size": 2,
        "fill_value": np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large integers with no common elements
    input_dict = {
        "ar1": np.array([1000, 2000], dtype=np.int64),
        "ar2": np.array([3000, 4000], dtype=np.int64),
        "assume_unique": True,
        "size": 3,
        "fill_value": np.array(-999, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.setdiff1d_1"] = setdiff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.setdiff1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.setdiff1d_1'.")


check_valid('jax.numpy.setdiff1d', generated_inputs['jax.numpy.setdiff1d_1'], lib="jax", suffix=1)
