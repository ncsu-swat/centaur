
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanquantile_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaNs, single quantile
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": np.array([0.5], dtype=np.float32),
            "axis": 0,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": False,
            "weights": weights,
        }
    )

    # Input 2: 2D array, multiple quantiles, axis 0, keepdims True
    a = np.array(
        [[10.0, np.nan, 30.0], [40.0, 50.0, 60.0], [np.nan, 80.0, 90.0]],
        dtype=np.float64,
    )
    weights = np.array(
        [[1.0, 1.0, 1.0], [2.0, 1.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float64
    )
    list_of_inputs.append(
        {
            "a": a,
            "q": np.array([0.25, 0.75], dtype=np.float64),
            "axis": 0,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": True,
            "weights": weights,
        }
    )

    # Input 3: 2D array, axis 1, keepdims False
    a = np.array(
        [[-5.0, -2.0, np.nan], [0.0, np.nan, 4.0], [-10.0, 2.0, 8.0]],
        dtype=np.float32,
    )
    weights = np.array(
        [[1.0, 1.0, 2.0], [1.0, 3.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32
    )
    list_of_inputs.append(
        {
            "a": a,
            "q": np.array([0.1, 0.5, 0.9], dtype=np.float32),
            "axis": 1,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": False,
            "weights": weights,
        }
    )

    # Input 4: Negative axis (-1) on 2D array
    a = np.array([[np.nan, 2.5, 3.5], [1.5, np.nan, 4.5]], dtype=np.float32)
    weights = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": np.array([0.0, 1.0], dtype=np.float32),
            "axis": -1,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": False,
            "weights": weights,
        }
    )

    # Input 5: 3D array with NaNs, axis 1
    a = np.array(
        [
            [[1.0, np.nan], [3.0, 4.0]],
            [[np.nan, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [np.nan, 12.0]],
        ],
        dtype=np.float32,
    )
    weights = np.ones((3, 2, 2), dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": np.array([0.5], dtype=np.float32),
            "axis": 1,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": True,
            "weights": weights,
        }
    )

    # Input 6: 3D array with negative values, axis 2
    a = np.array(
        [[[-1.0, -2.0, -3.0], [np.nan, -5.0, -6.0]]], dtype=np.float64
    )
    weights = np.array([[[1.0, 2.0, 1.0], [1.0, 1.0, 3.0]]], dtype=np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "q": np.array([0.2, 0.8], dtype=np.float64),
            "axis": 2,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": False,
            "weights": weights,
        }
    )

    # Input 7: Broadcastable weights along axis
    a = np.array(
        [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": np.array([0.5], dtype=np.float32),
            "axis": 1,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": False,
            "weights": weights,
        }
    )

    # Input 8: 1D array with negative values and float64
    a = np.array([-10.0, -5.0, np.nan, 0.0, 5.0, 10.0], dtype=np.float64)
    weights = np.array([0.5, 1.0, 1.0, 2.0, 1.0, 0.5], dtype=np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "q": np.array([0.25, 0.5, 0.75], dtype=np.float64),
            "axis": 0,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": True,
            "weights": weights,
        }
    )

    # Input 9: 4D array
    a = np.arange(16, dtype=np.float32).reshape((2, 2, 2, 2))
    a[0, 1, 0, 1] = np.nan
    weights = np.ones((2, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": np.array([0.5], dtype=np.float32),
            "axis": -2,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": False,
            "weights": weights,
        }
    )

    # Input 10: 2D array, quantile boundaries [0.0, 1.0], axis 0
    a = np.array(
        [[np.nan, 100.0], [200.0, np.nan], [300.0, 400.0]], dtype=np.float32
    )
    weights = np.array([[2.0, 1.0], [1.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "q": np.array([0.0, 0.5, 1.0], dtype=np.float32),
            "axis": 0,
            "overwrite_input": False,
            "method": "inverted_cdf",
            "keepdims": True,
            "weights": weights,
        }
    )

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
