
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanquantile_inputs():
    list_of_inputs = []

    # Helper function to create array with some NaNs
    def make_nan_array(shape, dtype=np.float32, nan_ratio=0.2):
        arr = np.random.randn(*shape).astype(dtype)
        mask = np.random.rand(*shape) < nan_ratio
        arr[mask] = np.nan
        return arr

    # Input 1: 1D array, q=0.5, axis=0
    a = make_nan_array((10,), np.float32)
    weights = (np.random.rand(10) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.5,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, q=0.25, axis=1, keepdims=True
    a = make_nan_array((4, 6), np.float32)
    weights = (np.random.rand(4, 6) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.25,
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, q=0.75, axis=2, float64
    a = make_nan_array((2, 3, 4), np.float64)
    weights = (np.random.rand(2, 3, 4) + 0.1).astype(np.float64)
    input_dict = {
        "a": a,
        "q": 0.75,
        "axis": 2,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array with no NaNs, q=0.0, axis=0
    a = (np.random.randn(5, 5)).astype(np.float32)
    weights = (np.random.rand(5, 5) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.0,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array with negative values, q=1.0, axis=0, keepdims=True
    a = make_nan_array((8,), np.float64) - 5.0
    weights = (np.random.rand(8) + 0.1).astype(np.float64)
    input_dict = {
        "a": a,
        "q": 1.0,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, q=0.1, axis=-1
    a = make_nan_array((3, 5), np.float32)
    weights = (np.random.rand(3, 5) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.1,
        "axis": -1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, q=0.9, axis=0, keepdims=True
    a = make_nan_array((3, 2, 3), np.float32)
    weights = (np.random.rand(3, 2, 3) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.9,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, q=0.3, axis=-1, float64
    a = make_nan_array((12,), np.float64)
    weights = (np.random.rand(12) + 0.1).astype(np.float64)
    input_dict = {
        "a": a,
        "q": 0.3,
        "axis": -1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, q=0.33, axis=0, keepdims=True
    a = make_nan_array((6, 4), np.float32)
    weights = (np.random.rand(6, 4) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.33,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, q=0.66, axis=1, float64
    a = make_nan_array((2, 4, 3), np.float64)
    weights = (np.random.rand(2, 4, 3) + 0.1).astype(np.float64)
    input_dict = {
        "a": a,
        "q": 0.66,
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanquantile_3"] = nanquantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanquantile_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanquantile_3'.")


check_valid('jax.numpy.nanquantile', generated_inputs['jax.numpy.nanquantile_3'], lib="jax", suffix=3)
