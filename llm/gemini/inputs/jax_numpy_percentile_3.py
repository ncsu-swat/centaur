
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def percentile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, median (q=50), axis=(0,), keepdims=False
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q = np.array(50.0, dtype=np.float32)
    axis = (0,)
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 2.0, 1.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": axis,
            "method": method,
            "keepdims": keepdims,
            "weights": weights,
        }
    )

    # Input 2: 1D int32 array, min/median/max percentiles, axis=(0,), keepdims=True
    a = np.array([-10, -5, 0, 5, 10, 15], dtype=np.int32)
    q = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    axis = (0,)
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": axis,
            "method": method,
            "keepdims": keepdims,
            "weights": weights,
        }
    )

    # Input 3: 2D float64 array with negative values, axis=(1,), keepdims=False
    a = np.array(
        [[-3.5, -1.2, 0.0], [2.1, -4.8, 6.3], [-0.5, 1.5, -2.5]],
        dtype=np.float64,
    )
    q = np.array([25.0, 75.0], dtype=np.float64)
    axis = (1,)
    method = "inverted_cdf"
    keepdims = False
    weights = np.array(
        [[1.0, 2.0, 1.0], [2.0, 1.0, 3.0], [1.0, 1.0, 2.0]], dtype=np.float64
    )
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": axis,
            "method": method,
            "keepdims": keepdims,
            "weights": weights,
        }
    )

    # Input 4: 2D float32 array, multiple quantiles, axis=(0,), keepdims=True
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    q = np.array([10.0, 50.0, 90.0], dtype=np.float32)
    axis = (0,)
    method = "inverted_cdf"
    keepdims = True
    weights = np.ones((4, 3), dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": axis,
            "method": method,
            "keepdims": keepdims,
            "weights": weights,
        }
    )

    # Input 5: 3D float32 array, single percentile, axis=(-1,), keepdims=False
    a = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    q = np.array([50.0], dtype=np.float32)
    axis = (-1,)
    method = "inverted_cdf"
    keepdims = False
    weights = np.random.uniform(0.5, 2.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": axis,
            "method": method,
            "keepdims": keepdims,
            "weights": weights,
        }
    )

    # Input 6: 3D int64 array, extremes (0 and 100), axis=(1,), keepdims=True
    a = np.array(
        [[[10, 20], [30, 40], [50, 60]], [[70, 80], [90, 100], [110, 120]]],
        dtype=np.int64,
    )
    q = np.array([0.0, 100.0], dtype=np.float64)
    axis = (1,)
    method = "inverted_cdf"
    keepdims = True
    weights = np.ones((2, 3, 2), dtype=np.int64)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": axis,
            "method": method,
            "keepdims": keepdims,
            "weights": weights,
        }
    )

    # Input 7: 2D float32 array, 0D scalar q array, axis=(-1,), keepdims=False
    a = np.array([[-10.0, -20.0, -30.0], [5.0, 15.0, 25.0]], dtype=np.float32)
    q = np.array(0.0, dtype=np.float32)
    axis = (-1,)
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([[1.0, 3.0, 2.0], [2.0, 1.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": axis,
            "method": method,
            "keepdims": keepdims,
            "weights": weights,
        }
    )

    # Input 8: 4D float32 array, axis=(2,), keepdims=True
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    q = np.array([5.0, 95.0], dtype=np.float32)
    axis = (2,)
    method = "inverted_cdf"
    keepdims = True
    weights = np.random.uniform(0.1, 1.0, size=(2, 3, 4, 5)).astype(np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": axis,
            "method": method,
            "keepdims": keepdims,
            "weights": weights,
        }
    )

    # Input 9: 1D float64 array, quartile boundaries, axis=(-1,), keepdims=False
    a = np.array(
        [100.5, 20.3, -4.2, 55.1, 88.9, -12.4, 0.0, 33.7], dtype=np.float64
    )
    q = np.array([12.5, 37.5, 62.5, 87.5], dtype=np.float64)
    axis = (-1,)
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.5, 2.0, 0.5, 1.0, 2.5, 1.0, 3.0, 0.5], dtype=np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": axis,
            "method": method,
            "keepdims": keepdims,
            "weights": weights,
        }
    )

    # Input 10: 3D float64 array, scalar q=100.0, axis=(0,), keepdims=False
    a = np.random.randn(5, 2, 3).astype(np.float64)
    q = np.array(100.0, dtype=np.float64)
    axis = (0,)
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones((5, 2, 3), dtype=np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": axis,
            "method": method,
            "keepdims": keepdims,
            "weights": weights,
        }
    )

    # Input 11: 2D int32 array, axis=(1,), keepdims=True
    a = np.array([[10, 2, 8, 4], [7, 1, 9, 3]], dtype=np.int32)
    q = np.array([20.0, 40.0, 60.0, 80.0], dtype=np.float32)
    axis = (1,)
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([[1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": axis,
            "method": method,
            "keepdims": keepdims,
            "weights": weights,
        }
    )

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
