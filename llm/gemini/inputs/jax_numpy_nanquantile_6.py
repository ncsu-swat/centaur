
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanquantile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with NaNs, median
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0, 1.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": np.array([0.5], dtype=np.float32),
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array with negative values and NaNs, multiple quantiles, keepdims=True
    a = np.array([-10.0, np.nan, -2.5, 0.0, 3.5, np.nan, 12.0], dtype=np.float64)
    weights = np.ones(a.shape, dtype=np.float64)
    input_dict = {
        "a": a,
        "q": np.array([0.25, 0.75], dtype=np.float64),
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, quantile along axis 0
    a = np.array(
        [
            [1.0, np.nan, 3.0],
            [np.nan, 5.0, 6.0],
            [7.0, 8.0, np.nan],
            [10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    weights = np.ones(a.shape, dtype=np.float32)
    input_dict = {
        "a": a,
        "q": np.array([0.5], dtype=np.float32),
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, quantiles along axis 1, keepdims=True
    a = np.array(
        [
            [-5.0, -1.0, np.nan, 4.0],
            [np.nan, 2.0, 3.0, 5.0],
            [0.0, 1.0, 2.0, np.nan],
        ],
        dtype=np.float64,
    )
    weights = np.array(
        [
            [1.0, 0.5, 1.0, 2.0],
            [1.0, 1.5, 0.5, 1.0],
            [2.0, 1.0, 1.0, 0.5],
        ],
        dtype=np.float64,
    )
    input_dict = {
        "a": a,
        "q": np.array([0.1, 0.9], dtype=np.float64),
        "axis": (1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, quantile along axis 2
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    weights = np.ones(a.shape, dtype=np.float32)
    input_dict = {
        "a": a,
        "q": np.array([0.5], dtype=np.float32),
        "axis": (2,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, negative axis index with multiple quantiles, keepdims=True
    a = np.random.default_rng(42).normal(size=(3, 4, 5)).astype(np.float32)
    a[a < -1.0] = np.nan
    weights = np.random.default_rng(42).uniform(0.1, 2.0, size=(3, 4, 5)).astype(np.float32)
    input_dict = {
        "a": a,
        "q": np.array([0.25, 0.5, 0.75], dtype=np.float32),
        "axis": (-1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 array, axis=-2
    a = np.array(
        [
            [-12.0, 0.0, np.nan],
            [3.0, np.nan, -1.0],
            [np.nan, 4.0, 2.0],
        ],
        dtype=np.float32,
    )
    weights = np.ones(a.shape, dtype=np.float32)
    input_dict = {
        "a": a,
        "q": np.array([0.3], dtype=np.float32),
        "axis": (-2,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, axis=1
    a = np.ones((2, 3, 2, 2), dtype=np.float32) * 5.0
    a[0, 0, 0, 0] = np.nan
    a[1, 2, 1, 1] = 10.0
    weights = np.ones(a.shape, dtype=np.float32)
    input_dict = {
        "a": a,
        "q": np.array([0.5], dtype=np.float32),
        "axis": (1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32 array, minimum quantile q=0.0
    a = np.array([np.nan, 10.0, 2.0, 50.0, np.nan, -3.0], dtype=np.float32)
    weights = np.array([1.0, 3.0, 2.0, 1.0, 1.0, 5.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "q": np.array([0.0], dtype=np.float32),
        "axis": (-1,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": False,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float64 array, maximum quantile q=1.0, axis=(0,), keepdims=True
    a = np.random.default_rng(123).uniform(-10.0, 10.0, size=(4, 3, 2)).astype(np.float64)
    a[0, 1, 0] = np.nan
    a[2, 0, 1] = np.nan
    weights = np.random.default_rng(123).uniform(0.5, 1.5, size=(4, 3, 2)).astype(np.float64)
    input_dict = {
        "a": a,
        "q": np.array([1.0], dtype=np.float64),
        "axis": (0,),
        "overwrite_input": False,
        "method": "inverted_cdf",
        "keepdims": True,
        "weights": weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.nanquantile_6"] = nanquantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanquantile_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanquantile_6'.")


check_valid('jax.numpy.nanquantile', generated_inputs['jax.numpy.nanquantile_6'], lib="jax", suffix=6)
