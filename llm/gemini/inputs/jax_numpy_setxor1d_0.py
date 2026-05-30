
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def setxor1d_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int arrays
    input_dict = {
        "ar1": np.array([1, 2, 3, 4], dtype=np.int32),
        "ar2": np.array([3, 4, 5, 6], dtype=np.int32),
        "assume_unique": False,
        "size": 4,
        "fill_value": np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Arrays with negative values
    input_dict = {
        "ar1": np.array([-1, -2, 0, 5], dtype=np.int32),
        "ar2": np.array([-2, 5, 10], dtype=np.int32),
        "assume_unique": False,
        "size": 5,
        "fill_value": np.array(-999, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float arrays, assumed unique
    input_dict = {
        "ar1": np.array([1.0, 2.0, 3.5], dtype=np.float32),
        "ar2": np.array([2.0, 3.5, 4.0], dtype=np.float32),
        "assume_unique": True,
        "size": 2,
        "fill_value": np.array(0.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays (flattened by setxor1d)
    input_dict = {
        "ar1": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "ar2": np.array([[3, 4], [5, 6]], dtype=np.int32),
        "assume_unique": False,
        "size": 6,
        "fill_value": np.array(-1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Arrays with duplicate values
    input_dict = {
        "ar1": np.array([1, 1, 2, 2, 3], dtype=np.int32),
        "ar2": np.array([2, 3, 4, 4], dtype=np.int32),
        "assume_unique": False,
        "size": 3,
        "fill_value": np.array(99, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8 arrays
    input_dict = {
        "ar1": np.array([10, 20, 30], dtype=np.uint8),
        "ar2": np.array([30, 40, 50], dtype=np.uint8),
        "assume_unique": True,
        "size": 4,
        "fill_value": np.array(0, dtype=np.uint8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 arrays with larger values
    input_dict = {
        "ar1": np.array([100000, 200000], dtype=np.int64),
        "ar2": np.array([200000, 300000], dtype=np.int64),
        "assume_unique": False,
        "size": 10,
        "fill_value": np.array(-1, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 arrays
    input_dict = {
        "ar1": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "ar2": np.array([0.3, 0.4, 0.5], dtype=np.float64),
        "assume_unique": True,
        "size": 3,
        "fill_value": np.array(0.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty array as the first input
    input_dict = {
        "ar1": np.array([], dtype=np.int32),
        "ar2": np.array([1, 2, 3], dtype=np.int32),
        "assume_unique": False,
        "size": 5,
        "fill_value": np.array(-1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with signed values and unique inputs
    input_dict = {
        "ar1": np.array([100, -100, 50, -50], dtype=np.int32),
        "ar2": np.array([50, -50, 25, -25], dtype=np.int32),
        "assume_unique": True,
        "size": 6,
        "fill_value": np.array(-999, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.setxor1d"] = setxor1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.setxor1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.setxor1d'.")


check_valid('jax.numpy.setxor1d', generated_inputs['jax.numpy.setxor1d'], lib="jax", suffix=0)
