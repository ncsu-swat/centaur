
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_quantile_inputs():
    list_of_inputs = []

    # Input 1: 1D array, median, simple weights
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q = 0.5
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, 25th percentile, keepdims=True
    a = np.random.randn(3, 4).astype(np.float32)
    q = 0.25
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.random.rand(3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, 75th percentile, float64, axis=(1,)
    a = np.random.randn(4, 4).astype(np.float64)
    q = 0.75
    axis = (1,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.rand(4, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=(0, 2), keepdims=True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    q = 0.9
    axis = (0, 2)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array with negative values, 10th percentile
    a = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0], dtype=np.float32)
    q = 0.1
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.rand(10).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, min value (q=0.0)
    a = np.random.randn(5, 2).astype(np.float32)
    q = 0.0
    axis = (1,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.rand(5, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, max value (q=1.0), axis=(1, 2)
    a = np.random.randn(3, 3, 3).astype(np.float32)
    q = 1.0
    axis = (1, 2)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.random.rand(3, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, axis=(0, 1, 2, 3)
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    q = 0.5
    axis = (0, 1, 2, 3)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, weights matching exact shape of a
    a = np.random.randn(10, 5).astype(np.float32)
    q = 0.3
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.rand(10, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, large size, q=0.6, keepdims=True
    a = np.random.randn(100).astype(np.float32)
    q = 0.6
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.random.rand(100).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "overwrite_input": overwrite_input,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.quantile_4"] = jax_numpy_quantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.quantile_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.quantile_4'.")


check_valid('jax.numpy.quantile', generated_inputs['jax.numpy.quantile_4'], lib="jax", suffix=4)
