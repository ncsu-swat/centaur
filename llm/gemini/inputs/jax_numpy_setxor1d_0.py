
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def setxor1d_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D int32 arrays with size and fill_value
    input_dict = {
        "ar1": np.array([1, 2, 3, 4], dtype=np.int32),
        "ar2": np.array([3, 4, 5, 6], dtype=np.int32),
        "assume_unique": False,
        "size": 6,
        "fill_value": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 arrays with negative numbers
    input_dict = {
        "ar1": np.array([-2.5, -1.0, 0.0, 1.5], dtype=np.float32),
        "ar2": np.array([-1.0, 1.5, 2.0, 3.5], dtype=np.float32),
        "assume_unique": False,
        "size": 5,
        "fill_value": np.array(-99.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional arrays (flattened by setxor1d)
    input_dict = {
        "ar1": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "ar2": np.array([[3, 4], [5, 6]], dtype=np.int64),
        "assume_unique": False,
        "size": 4,
        "fill_value": np.array(-1, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: assume_unique=True with unique elements in each input
    input_dict = {
        "ar1": np.array([10, 20, 30, 40], dtype=np.int32),
        "ar2": np.array([30, 40, 50, 60], dtype=np.int32),
        "assume_unique": True,
        "size": 4,
        "fill_value": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Duplicate values in input arrays with assume_unique=False
    input_dict = {
        "ar1": np.array([1, 1, 2, 2, 3], dtype=np.int32),
        "ar2": np.array([2, 3, 3, 4, 4], dtype=np.int32),
        "assume_unique": False,
        "size": 3,
        "fill_value": np.array(-1, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean arrays
    input_dict = {
        "ar1": np.array([True, False, True], dtype=np.bool_),
        "ar2": np.array([False, False], dtype=np.bool_),
        "assume_unique": False,
        "size": 2,
        "fill_value": np.array(False, dtype=np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 arrays with size larger than result (padding required)
    input_dict = {
        "ar1": np.array([0.5, 1.5, 2.5], dtype=np.float64),
        "ar2": np.array([1.5, 3.5], dtype=np.float64),
        "assume_unique": True,
        "size": 8,
        "fill_value": np.array(np.nan, dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty array inputs
    input_dict = {
        "ar1": np.array([], dtype=np.int32),
        "ar2": np.array([1, 2, 3], dtype=np.int32),
        "assume_unique": False,
        "size": 5,
        "fill_value": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Disjoint sets
    input_dict = {
        "ar1": np.array([1, 3, 5, 7], dtype=np.int32),
        "ar2": np.array([2, 4, 6, 8], dtype=np.int32),
        "assume_unique": True,
        "size": 8,
        "fill_value": np.array(-1, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Completely identical inputs (empty symmetric difference)
    input_dict = {
        "ar1": np.array([1, 2, 3, 4], dtype=np.int64),
        "ar2": np.array([1, 2, 3, 4], dtype=np.int64),
        "assume_unique": True,
        "size": 4,
        "fill_value": np.array(0, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D input arrays
    input_dict = {
        "ar1": np.arange(8, dtype=np.int32).reshape(2, 2, 2),
        "ar2": np.arange(4, 12, dtype=np.int32).reshape(2, 2, 2),
        "assume_unique": False,
        "size": 10,
        "fill_value": np.array(999, dtype=np.int32),
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
