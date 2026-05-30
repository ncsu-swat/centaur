
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def percentile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, 0D q, axis=(0,), keepdims=False
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q = np.array(50.0, dtype=np.float32)
    axis = (0,)
    method = 'inverted_cdf'
    keepdims = False
    weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 2: 1D int32 array, 1D q, axis=(0,), keepdims=True
    a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int32)
    q = np.array([25.0, 75.0], dtype=np.float32)
    axis = (0,)
    method = 'inverted_cdf'
    keepdims = True
    weights = np.ones((10,), dtype=np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 3: 2D float64 array, 0D q, axis=(0,), keepdims=False
    a = np.random.randn(3, 4).astype(np.float64)
    q = np.array(90.0, dtype=np.float64)
    axis = (0,)
    method = 'inverted_cdf'
    keepdims = False
    weights = np.random.rand(3, 4).astype(np.float64) + 0.1
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 4: 2D float32 array with negative values, 1D q, axis=(1,), keepdims=True
    a = np.random.randn(4, 3).astype(np.float32)
    q = np.array([10.0, 90.0], dtype=np.float32)
    axis = (1,)
    method = 'inverted_cdf'
    keepdims = True
    weights = np.random.rand(4, 3).astype(np.float32) + 0.1
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 5: 3D float32 array, 0D q, axis=(0, 2), keepdims=False
    a = np.random.randn(2, 3, 4).astype(np.float32)
    q = np.array(25.5, dtype=np.float32)
    axis = (0, 2)
    method = 'inverted_cdf'
    keepdims = False
    weights = np.random.rand(2, 3, 4).astype(np.float32) + 0.1
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 6: 2D int32 array, 1D q, axis=(-1,), keepdims=True
    a = np.random.randint(-100, 100, size=(5, 5)).astype(np.int32)
    q = np.array([50.0], dtype=np.float32)
    axis = (-1,)
    method = 'inverted_cdf'
    keepdims = True
    weights = np.random.rand(5, 5).astype(np.float32) + 0.5
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 7: Large 1D float64 array, 1D q, axis=(0,), keepdims=False
    a = np.random.randn(100).astype(np.float64)
    q = np.array([5.0, 25.0, 50.0, 75.0, 95.0], dtype=np.float64)
    axis = (0,)
    method = 'inverted_cdf'
    keepdims = False
    weights = np.random.rand(100).astype(np.float64) + 0.1
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 8: 3D float32 array, 0D q (0th percentile), axis=(1, 2), keepdims=True
    a = np.random.randn(3, 2, 2).astype(np.float32)
    q = np.array(0.0, dtype=np.float32)
    axis = (1, 2)
    method = 'inverted_cdf'
    keepdims = True
    weights = np.random.rand(3, 2, 2).astype(np.float32) + 0.1
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 9: 2D float32 array, 0D q (100th percentile), axis=(0,), keepdims=False
    a = np.random.randn(6, 2).astype(np.float32)
    q = np.array(100.0, dtype=np.float32)
    axis = (0,)
    method = 'inverted_cdf'
    keepdims = False
    weights = np.random.rand(6, 2).astype(np.float32) + 0.1
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 10: 4D float32 array, 1D q, axis=(2, 3), keepdims=True
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    q = np.array([30.0, 70.0], dtype=np.float32)
    axis = (2, 3)
    method = 'inverted_cdf'
    keepdims = True
    weights = np.random.rand(2, 2, 3, 3).astype(np.float32) + 0.1
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "method": method, "keepdims": keepdims, "weights": weights
    })

    return list_of_inputs

generated_inputs["jax.numpy.percentile_3"] = percentile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.percentile_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.percentile_3'.")


check_valid('jax.numpy.percentile', generated_inputs['jax.numpy.percentile_3'], lib="jax", suffix=3)
