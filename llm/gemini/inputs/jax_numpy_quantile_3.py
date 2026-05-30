
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_quantile_inputs():
    list_of_inputs = []

    # Input 1: 1D float array, median
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "q": 0.5,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    })

    # Input 2: 2D array, 25th percentile along axis 1, keepdims=True
    a = np.random.randn(3, 4).astype(np.float32)
    weights = np.random.uniform(0.5, 1.5, size=(3, 4)).astype(np.float32)
    list_of_inputs.append({
        "a": a,
        "q": 0.25,
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    })

    # Input 3: 3D float64 array, 75th percentile along last axis
    a = np.random.randn(2, 3, 2).astype(np.float64)
    weights = np.random.uniform(0.1, 2.0, size=(2, 3, 2)).astype(np.float64)
    list_of_inputs.append({
        "a": a,
        "q": 0.75,
        "axis": -1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    })

    # Input 4: 1D array with negative values, 10th percentile
    a = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    weights = np.array([0.5, 1.5, 1.0, 1.0, 0.5], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "q": 0.1,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    })

    # Input 5: 2D array, 90th percentile along axis 0
    a = np.random.randint(0, 100, size=(5, 5)).astype(np.float32)
    weights = np.random.uniform(1.0, 5.0, size=(5, 5)).astype(np.float32)
    list_of_inputs.append({
        "a": a,
        "q": 0.9,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    })

    # Input 6: 4D array, median along axis 2, keepdims=True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    weights = np.random.uniform(0.1, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({
        "a": a,
        "q": 0.5,
        "axis": 2,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    })

    # Input 7: Single element array, 0th percentile
    a = np.array([42.0], dtype=np.float32)
    weights = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "q": 0.0,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    })

    # Input 8: 2D array with large values, 100th percentile
    a = np.random.randn(4, 3).astype(np.float32) * 100.0
    weights = np.random.uniform(0.5, 2.0, size=(4, 3)).astype(np.float32)
    list_of_inputs.append({
        "a": a,
        "q": 1.0,
        "axis": -2,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    })

    # Input 9: 3D float64 array, 33rd percentile along axis 1
    a = np.random.uniform(-1.0, 1.0, size=(3, 2, 3)).astype(np.float64)
    weights = np.ones((3, 2, 3), dtype=np.float64)
    list_of_inputs.append({
        "a": a,
        "q": 0.33,
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    })

    # Input 10: 1D binary-like array, median with custom weights
    a = np.array([0, 1, 0, 1, 1, 0], dtype=np.float32)
    weights = np.array([1, 2, 1, 2, 1, 2], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "q": 0.5,
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    })

    return list_of_inputs

generated_inputs["jax.numpy.quantile_3"] = jax_numpy_quantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.quantile_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.quantile_3'.")


check_valid('jax.numpy.quantile', generated_inputs['jax.numpy.quantile_3'], lib="jax", suffix=3)
