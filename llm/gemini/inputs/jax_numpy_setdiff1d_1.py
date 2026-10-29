
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def setdiff1d_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D int32 arrays with padding (size > result length)
    input_dict = {
        "ar1": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "ar2": np.array([2, 4, 6], dtype=np.int32),
        "assume_unique": False,
        "size": 5,
        "fill_value": np.array(-1, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Unique int64 arrays with negative values
    input_dict = {
        "ar1": np.array([-10, -5, 0, 5, 10], dtype=np.int64),
        "ar2": np.array([-5, 5], dtype=np.int64),
        "assume_unique": True,
        "size": 3,
        "fill_value": np.array(0, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float32 arrays with truncation (size < result length)
    input_dict = {
        "ar1": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        "ar2": np.array([0.5, 5.5], dtype=np.float32),
        "assume_unique": False,
        "size": 2,
        "fill_value": np.array(-999.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float64 arrays with negative and positive values, exact size
    input_dict = {
        "ar1": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64),
        "ar2": np.array([0.0, 2.0], dtype=np.float64),
        "assume_unique": True,
        "size": 3,
        "fill_value": np.array(0.0, dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int32 arrays with duplicates and assume_unique=False
    input_dict = {
        "ar1": np.array([10, 10, 20, 20, 30, 40], dtype=np.int32),
        "ar2": np.array([20, 40, 50], dtype=np.int32),
        "assume_unique": False,
        "size": 4,
        "fill_value": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32 arrays that get flattened
    input_dict = {
        "ar1": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "ar2": np.array([[2, 3], [5, 6]], dtype=np.int32),
        "assume_unique": False,
        "size": 4,
        "fill_value": np.array(-1, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Fully overlapping arrays (all ar1 elements in ar2)
    input_dict = {
        "ar1": np.array([10, 20, 30], dtype=np.int32),
        "ar2": np.array([10, 20, 30, 40], dtype=np.int32),
        "assume_unique": True,
        "size": 3,
        "fill_value": np.array(999, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64 arrays with negative values and duplicates
    input_dict = {
        "ar1": np.array([-50, -50, 0, 50, 100], dtype=np.int64),
        "ar2": np.array([-50, 100, 100], dtype=np.int64),
        "assume_unique": False,
        "size": 5,
        "fill_value": np.array(-1, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty second array (ar2 is empty)
    input_dict = {
        "ar1": np.array([7, 8, 9], dtype=np.int64),
        "ar2": np.array([], dtype=np.int64),
        "assume_unique": True,
        "size": 4,
        "fill_value": np.array(-1, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Size is 1 (minimal size truncation)
    input_dict = {
        "ar1": np.array([100, 200, 300, 400], dtype=np.int32),
        "ar2": np.array([50, 150], dtype=np.int32),
        "assume_unique": False,
        "size": 1,
        "fill_value": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Float32 arrays with multiple duplicates
    input_dict = {
        "ar1": np.array([0.0, 1.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "ar2": np.array([1.0, 4.0], dtype=np.float32),
        "assume_unique": False,
        "size": 5,
        "fill_value": np.array(-1.0, dtype=np.float32),
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
