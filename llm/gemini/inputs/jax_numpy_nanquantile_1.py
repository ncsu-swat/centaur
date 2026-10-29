
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanquantile_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaNs, scalar q, axis=0
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
    q = np.array(0.5, dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": 0,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": False,
            "weights": weights,
        }
    )

    # Input 2: 1D array with multiple quantiles, keepdims=True
    a = np.array([10.0, np.nan, 30.0, 40.0, 50.0, 60.0], dtype=np.float32)
    q = np.array([0.25, 0.75], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.5, 0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": 0,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": True,
            "weights": weights,
        }
    )

    # Input 3: 2D array, axis=0, float64
    a = np.array(
        [[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]],
        dtype=np.float64,
    )
    q = np.array(0.5, dtype=np.float64)
    weights = np.ones((3, 3), dtype=np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": 0,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": False,
            "weights": weights,
        }
    )

    # Input 4: 2D array, axis=1, multiple quantiles, keepdims=True
    a = np.array(
        [
            [-5.0, -2.0, np.nan, 1.0],
            [0.0, np.nan, 4.0, 6.0],
            [np.nan, -1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    q = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    weights = np.array(
        [[1.0, 2.0, 1.0, 1.0], [2.0, 1.0, 3.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": 1,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": True,
            "weights": weights,
        }
    )

    # Input 5: 2D array, axis=-1, negative values, keepdims=False
    a = np.array(
        [[-10.0, -20.0, np.nan], [-30.0, np.nan, -50.0]], dtype=np.float32
    )
    q = np.array(0.0, dtype=np.float32)
    weights = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": -1,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": False,
            "weights": weights,
        }
    )

    # Input 6: 3D array, axis=0, keepdims=False
    a = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    q = np.array(0.7, dtype=np.float32)
    weights = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": 0,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": False,
            "weights": weights,
        }
    )

    # Input 7: 3D array, axis=2, keepdims=True
    a = np.array(
        [
            [[np.nan, 2.0], [3.0, 4.0]],
            [[5.0, np.nan], [7.0, 8.0]],
            [[9.0, 10.0], [np.nan, 12.0]],
        ],
        dtype=np.float64,
    )
    q = np.array([0.2, 0.8], dtype=np.float64)
    weights = np.ones((3, 2, 2), dtype=np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": 2,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": True,
            "weights": weights,
        }
    )

    # Input 8: 1D array, edge quantiles (q=1.0), axis=-1
    a = np.array([np.nan, 3.14, -2.71, 0.0, np.nan, 1.41], dtype=np.float32)
    q = np.array(1.0, dtype=np.float32)
    weights = np.array([0.5, 1.5, 2.0, 1.0, 0.5, 3.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": -1,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": False,
            "weights": weights,
        }
    )

    # Input 9: 4D array, axis=1, keepdims=True
    a = np.ones((2, 3, 2, 2), dtype=np.float32)
    a[0, 0, 0, 0] = np.nan
    a[1, 1, 1, 0] = 5.0
    q = np.array(0.3, dtype=np.float32)
    weights = np.ones((2, 3, 2, 2), dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": 1,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": True,
            "weights": weights,
        }
    )

    # Input 10: 2D array, axis=1, 1D quantile array [0.0, 0.5, 1.0], keepdims=False
    a = np.array(
        [[100.0, np.nan, 300.0], [400.0, 500.0, 600.0]], dtype=np.float32
    )
    q = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    weights = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": q,
            "axis": 1,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": False,
            "weights": weights,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.nanquantile_1"] = nanquantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanquantile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanquantile_1'.")


check_valid('jax.numpy.nanquantile', generated_inputs['jax.numpy.nanquantile_1'], lib="jax", suffix=1)
