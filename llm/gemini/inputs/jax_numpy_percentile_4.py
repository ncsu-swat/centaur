
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def percentile_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array, median
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q = 50.0
    axis = (0,)
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, reduction along axis 1 with keepdims
    a = np.random.randn(3, 4).astype(np.float32)
    q = 25.0
    axis = (1,)
    method = "inverted_cdf"
    keepdims = True
    weights = np.ones((3, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, reduction along axis 0
    a = np.random.randn(4, 3).astype(np.float64)
    q = 75.0
    axis = (0,)
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones((4, 3), dtype=np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, reduction along axis 2 with keepdims
    a = np.random.randn(2, 3, 4).astype(np.float32)
    q = 90.0
    axis = (2,)
    method = "inverted_cdf"
    keepdims = True
    weights = np.ones((2, 3, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array with negative and positive values, low quantile
    a = np.array([-10.0, 0.0, 10.0, 20.0], dtype=np.float32)
    q = 10.0
    axis = (0,)
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, reduction along multiple axes (0, 1)
    a = np.random.randn(3, 3).astype(np.float32)
    q = 50.0
    axis = (0, 1)
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.rand(3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, 0th percentile (minimum)
    a = np.array([5.0, 3.0, 9.0, 1.0], dtype=np.float32)
    q = 0.0
    axis = (0,)
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float64 array, 100th percentile (maximum)
    a = np.array([5.0, 3.0, 9.0, 1.0], dtype=np.float64)
    q = 100.0
    axis = (0,)
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, reduction along axes (0, 2) with keepdims
    a = np.random.randn(2, 2, 2).astype(np.float32)
    q = 33.3
    axis = (0, 2)
    method = "inverted_cdf"
    keepdims = True
    weights = np.ones((2, 2, 2), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, custom weights, reduction along axis 1
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    q = 60.0
    axis = (1,)
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.percentile_4"] = percentile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.percentile_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.percentile_4'.")


check_valid('jax.numpy.percentile', generated_inputs['jax.numpy.percentile_4'], lib="jax", suffix=4)
