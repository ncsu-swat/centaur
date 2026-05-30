
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanquantile_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array with weights
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "q": np.array([0.5], dtype=np.float32),
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    })

    # Input 2: 2D array, computing along axis 1
    a = np.array([[np.nan, 2.0, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    weights = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "q": np.array([0.25, 0.75], dtype=np.float32),
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    })

    # Input 3: Float64 2D array, computing along axis 0
    a = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float64)
    weights = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    list_of_inputs.append({
        "a": a,
        "q": np.array([0.5], dtype=np.float64),
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    })

    # Input 4: Larger 1D array with multiple quantiles and keepdims=True
    a = np.random.randn(10).astype(np.float32)
    a[3] = np.nan
    weights = np.abs(np.random.randn(10).astype(np.float32))
    list_of_inputs.append({
        "a": a,
        "q": np.array([0.1, 0.5, 0.9], dtype=np.float32),
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    })

    # Input 5: 2D array, computing along axis 0
    a = np.random.randn(4, 3).astype(np.float32)
    a[1, 1] = np.nan
    weights = np.abs(np.random.randn(4, 3).astype(np.float32))
    list_of_inputs.append({
        "a": a,
        "q": np.array([0.5], dtype=np.float32),
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    })

    # Input 6: 3D array computing along axis 2
    a = np.random.randn(2, 3, 2).astype(np.float32)
    a[0, 0, 0] = np.nan
    weights = np.ones((2, 3, 2), dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "q": np.array([0.2, 0.8], dtype=np.float32),
        "axis": 2,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    })

    # Input 7: Float64 2D array, negative axis
    a = np.random.randn(5, 5).astype(np.float64)
    a[2, 3] = np.nan
    weights = np.ones((5, 5), dtype=np.float64)
    list_of_inputs.append({
        "a": a,
        "q": np.array([0.5], dtype=np.float64),
        "axis": -1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    })

    # Input 8: 2D array, computing along axis 1
    a = np.random.randn(3, 4).astype(np.float32)
    a[0, 0] = np.nan
    weights = np.ones((3, 4), dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "q": np.array([0.25, 0.5], dtype=np.float32),
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights
    })

    # Input 9: 1D array with 0.0 and 1.0 quantiles
    a = np.random.randn(8).astype(np.float32)
    weights = np.ones((8,), dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "q": np.array([0.0, 1.0], dtype=np.float32),
        "axis": 0,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    })

    # Input 10: 3D array computing along axis 1
    a = np.random.randn(2, 2, 2).astype(np.float32)
    a[1, 1, 1] = np.nan
    weights = np.ones((2, 2, 2), dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "q": np.array([0.75], dtype=np.float32),
        "axis": 1,
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights
    })

    return list_of_inputs

generated_inputs["jax.numpy.nanquantile_5"] = nanquantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanquantile_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanquantile_5'.")


check_valid('jax.numpy.nanquantile', generated_inputs['jax.numpy.nanquantile_5'], lib="jax", suffix=5)
