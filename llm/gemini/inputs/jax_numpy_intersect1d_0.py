
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def intersect1d_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D integer arrays
    input_dict = {
        "ar1": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "ar2": np.array([3, 4, 5, 6, 7], dtype=np.int32),
        "assume_unique": False,
        "return_indices": False,
        "size": 3,
        "fill_value": np.int32(0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float arrays with negative values and return_indices=True
    input_dict = {
        "ar1": np.array([-2.5, -1.0, 0.0, 1.5, 3.0], dtype=np.float32),
        "ar2": np.array([-1.0, 1.5, 4.5], dtype=np.float32),
        "assume_unique": True,
        "return_indices": True,
        "size": 2,
        "fill_value": np.float32(-999.0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional arrays (flattened by intersect1d)
    input_dict = {
        "ar1": np.array([[10, 20], [30, 40]], dtype=np.int64),
        "ar2": np.array([[20, 50], [40, 60]], dtype=np.int64),
        "assume_unique": False,
        "return_indices": True,
        "size": 4,
        "fill_value": np.int64(-1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Size larger than intersection count (padding with fill_value)
    input_dict = {
        "ar1": np.array([1, 2, 3], dtype=np.int32),
        "ar2": np.array([2, 4, 6], dtype=np.int32),
        "assume_unique": False,
        "return_indices": False,
        "size": 5,
        "fill_value": np.int32(99),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Size smaller than intersection count (truncation)
    input_dict = {
        "ar1": np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
        "ar2": np.array([2, 3, 4, 5, 6, 7], dtype=np.int32),
        "assume_unique": True,
        "return_indices": False,
        "size": 2,
        "fill_value": np.int32(0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Double precision float inputs with duplicates
    input_dict = {
        "ar1": np.array([1.1, 2.2, 2.2, 3.3, 4.4], dtype=np.float64),
        "ar2": np.array([2.2, 3.3, 3.3, 5.5], dtype=np.float64),
        "assume_unique": False,
        "return_indices": True,
        "size": 3,
        "fill_value": np.float64(0.0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean arrays
    input_dict = {
        "ar1": np.array([True, False, True], dtype=np.bool_),
        "ar2": np.array([False, False], dtype=np.bool_),
        "assume_unique": False,
        "return_indices": False,
        "size": 2,
        "fill_value": np.bool_(False),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty array intersection
    input_dict = {
        "ar1": np.array([], dtype=np.int32),
        "ar2": np.array([1, 2, 3], dtype=np.int32),
        "assume_unique": True,
        "return_indices": True,
        "size": 3,
        "fill_value": np.int32(-1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Disjoint sets with zero intersection
    input_dict = {
        "ar1": np.array([100, 200, 300], dtype=np.int32),
        "ar2": np.array([400, 500, 600], dtype=np.int32),
        "assume_unique": True,
        "return_indices": False,
        "size": 3,
        "fill_value": np.int32(0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array inputs with size equal to 0
    input_dict = {
        "ar1": np.arange(8, dtype=np.int32).reshape(2, 2, 2),
        "ar2": np.arange(4, 12, dtype=np.int32).reshape(2, 4),
        "assume_unique": False,
        "return_indices": True,
        "size": 0,
        "fill_value": np.int32(0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.intersect1d"] = intersect1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.intersect1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.intersect1d'.")


check_valid('jax.numpy.intersect1d', generated_inputs['jax.numpy.intersect1d'], lib="jax", suffix=0)
