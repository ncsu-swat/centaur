
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanquantile_inputs():
    list_of_inputs = []

    # Case 1: 1D array, basic float32
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.5,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D array, reduction along axis 0
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]], dtype=np.float32)
    weights = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.25,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D array, reduction along axis 1 with negative numbers
    a = np.array([[-10.0, np.nan, -3.0], [4.0, -5.0, np.nan]], dtype=np.float32)
    weights = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.75,
        "axis": (1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D array, float64, reduction along axis 1
    a = np.random.uniform(-10, 10, size=(2, 3, 4)).astype(np.float64)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    weights = np.random.uniform(0.5, 2.0, size=(2, 3, 4)).astype(np.float64)
    input_dict = {
        "a": a,
        "q": 0.1,
        "axis": (1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 3D array, reduction along axis 2, keepdims=True
    a = np.random.uniform(0, 100, size=(3, 2, 5)).astype(np.float32)
    a[2, 1, 4] = np.nan
    weights = np.random.uniform(1.0, 5.0, size=(3, 2, 5)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.9,
        "axis": (2,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 1D array with single element
    a = np.array([42.0], dtype=np.float32)
    weights = np.array([1.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.5,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: 4D array with small dimensions
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    a[0, 1, 0, 1] = np.nan
    weights = np.random.uniform(0.1, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.3,
        "axis": (3,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: 1D array, q = 0.0 (min)
    a = np.array([np.nan, -1.0, -5.0, 10.0, np.nan], dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 0.0,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: 1D array, q = 1.0 (max)
    a = np.array([15.0, np.nan, 3.0, np.nan, 50.0], dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": 1.0,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: 2D array, large size with various NaNs
    a = np.random.uniform(-100, 100, size=(10, 10)).astype(np.float32)
    a[a < 0] = np.nan  # Introduce NaNs
    weights = np.random.uniform(0.5, 1.5, size=(10, 10)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": 0.6,
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanquantile_4"] = nanquantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanquantile_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanquantile_4'.")


check_valid('jax.numpy.nanquantile', generated_inputs['jax.numpy.nanquantile_4'], lib="jax", suffix=4)
