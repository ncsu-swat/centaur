
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def percentile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, simple percentile
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q = np.array([25.0, 50.0, 75.0], dtype=np.float32)
    axis = 0
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

    # Input 2: 1D int32 array, keepdims=True
    a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int32)
    q = np.array([50.0], dtype=np.float32)
    axis = 0
    method = "inverted_cdf"
    keepdims = True
    weights = np.ones((10,), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, axis=0
    a = np.random.uniform(0.0, 100.0, size=(4, 3)).astype(np.float32)
    q = np.array([10.0, 90.0], dtype=np.float32)
    axis = 0
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.uniform(0.1, 1.0, size=(4, 3)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, axis=1, keepdims=True
    a = np.random.uniform(-50.0, 50.0, size=(3, 5)).astype(np.float64)
    q = np.array([20.0, 40.0, 60.0, 80.0], dtype=np.float64)
    axis = 1
    method = "inverted_cdf"
    keepdims = True
    weights = np.random.uniform(0.5, 2.0, size=(3, 5)).astype(np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, axis=2
    a = np.random.normal(0.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    q = np.array([5.0, 50.0, 95.0], dtype=np.float32)
    axis = 2
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.uniform(0.1, 1.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D negative float32 array
    a = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0], dtype=np.float32)
    q = np.array([30.0], dtype=np.float32)
    axis = 0
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int32 array with negative values, axis=-1
    a = np.array([
        [-10, -20, -30, -40, -50],
        [10, 20, 30, 40, 50],
        [-5, -2, 0, 2, 5],
        [100, 200, 300, 400, 500],
        [-100, -200, -300, -400, -500]
    ], dtype=np.int32)
    q = np.array([20.0, 50.0, 80.0], dtype=np.float32)
    axis = -1
    method = "inverted_cdf"
    keepdims = True
    weights = np.ones((5, 5), dtype=np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64 array, axis=-2
    a = np.random.rand(2, 2, 2).astype(np.float64)
    q = np.array([10.0, 90.0], dtype=np.float64)
    axis = -2
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.rand(2, 2, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with 5 percentiles, keepdims=True
    a = np.random.randn(12).astype(np.float32)
    q = np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype=np.float32)
    axis = 0
    method = "inverted_cdf"
    keepdims = True
    weights = np.random.uniform(0.1, 1.0, size=(12,)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": q,
        "axis": axis,
        "method": method,
        "keepdims": keepdims,
        "weights": weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, axis=0, float32, keepdims=False
    a = np.random.randn(6, 2).astype(np.float32)
    q = np.array([50.0], dtype=np.float32)
    axis = 0
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.uniform(0.5, 1.5, size=(6, 2)).astype(np.float32)
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

generated_inputs["jax.numpy.percentile_1"] = percentile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.percentile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.percentile_1'.")


check_valid('jax.numpy.percentile', generated_inputs['jax.numpy.percentile_1'], lib="jax", suffix=1)
