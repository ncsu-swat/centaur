
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def percentile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard case
    a = np.random.randn(10).astype(np.float32)
    q = 50.0
    axis = 0
    method = "inverted_cdf"
    keepdims = False
    weights = (np.random.rand(10) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, axis 1, keepdims True
    a = np.random.randn(5, 5).astype(np.float32)
    q = 25.0
    axis = 1
    method = "inverted_cdf"
    keepdims = True
    weights = (np.random.rand(5, 5) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32 array, axis 2, keepdims False
    a = np.random.randn(2, 3, 4).astype(np.float32)
    q = 75.0
    axis = 2
    method = "inverted_cdf"
    keepdims = False
    weights = (np.random.rand(2, 3, 4) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float64 array with negative values, q=10.0, keepdims True
    a = (np.random.randn(15) - 5.0).astype(np.float64)
    q = 10.0
    axis = 0
    method = "inverted_cdf"
    keepdims = True
    weights = (np.random.rand(15) + 0.1).astype(np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int32 array, axis 0, float32 weights
    a = np.random.randint(-100, 100, size=(3, 6)).astype(np.int32)
    q = 90.0
    axis = 0
    method = "inverted_cdf"
    keepdims = False
    weights = (np.random.rand(3, 6) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, axis 3, q=0.0
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    q = 0.0
    axis = 3
    method = "inverted_cdf"
    keepdims = True
    weights = (np.random.rand(2, 2, 2, 2) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array, negative axis, q=100.0
    a = np.random.randn(8).astype(np.float32)
    q = 100.0
    axis = -1
    method = "inverted_cdf"
    keepdims = False
    weights = (np.random.rand(8) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64 array, axis 1, float64 weights
    a = np.random.randn(4, 2, 3).astype(np.float64)
    q = 33.3
    axis = 1
    method = "inverted_cdf"
    keepdims = False
    weights = (np.random.rand(4, 2, 3) + 0.1).astype(np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, axis -2, keepdims True
    a = np.random.randn(10, 2).astype(np.float32)
    q = 66.6
    axis = -2
    method = "inverted_cdf"
    keepdims = True
    weights = (np.random.rand(10, 2) + 0.1).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large 1D int32 array, q=50.0
    a = np.arange(100).astype(np.int32)
    q = 50.0
    axis = 0
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones(100).astype(np.float32)
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

generated_inputs["jax.numpy.percentile_2"] = percentile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.percentile_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.percentile_2'.")


check_valid('jax.numpy.percentile', generated_inputs['jax.numpy.percentile_2'], lib="jax", suffix=2)
