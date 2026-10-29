
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanpercentile_inputs():
    list_of_inputs = []

    # Input 1: 1D array, single percentile, axis=(0,)
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
    q = np.array(50.0, dtype=np.float32)
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 1.0, 1.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 2: 1D array, multiple percentiles, float64
    a = np.array([np.nan, 10.0, 20.0, 30.0, 40.0, np.nan], dtype=np.float64)
    q = np.array([25.0, 75.0], dtype=np.float64)
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([1.0, 2.0, 1.0, 1.0, 2.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 3: 2D array, axis=(0,), keepdims=False
    a = np.array(
        [[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    q = np.array(50.0, dtype=np.float32)
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones((3, 3), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 4: 2D array, axis=(1,), keepdims=True, negative values
    a = np.array(
        [[-10.0, -5.0, np.nan], [np.nan, 0.0, 15.0]], dtype=np.float32
    )
    q = np.array([0.0, 100.0], dtype=np.float32)
    axis = (1,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 3.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 5: 2D array, multi-axis reduction axis=(0, 1)
    a = np.array(
        [[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0]],
        dtype=np.float32,
    )
    q = np.array(75.0, dtype=np.float32)
    axis = (0, 1)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones((2, 3), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 6: 3D array, axis=(-1,), varying weights
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    q = np.array([10.0, 50.0, 90.0], dtype=np.float32)
    axis = (-1,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 7: 3D array, axis=(0, 2), keepdims=True
    a = np.array(
        [[[1.0, np.nan], [3.0, 4.0]], [[np.nan, 6.0], [7.0, 8.0]]],
        dtype=np.float64,
    )
    q = np.array(30.0, dtype=np.float64)
    axis = (0, 2)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.ones((2, 2, 2), dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 8: 1D array with all non-NaN values
    a = np.array([5.0, 2.0, 9.0, 1.0, 7.0], dtype=np.float32)
    q = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 2.0, 0.5, 1.5, 3.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 9: 4D array, axis=(2,), keepdims=False
    a = np.zeros((2, 2, 3, 2), dtype=np.float32)
    a[0, 0, 1, 0] = np.nan
    q = np.array(25.0, dtype=np.float32)
    axis = (2,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones((2, 2, 3, 2), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    # Input 10: 2D array with negative axis=(-2,), float64
    a = np.array(
        [[-3.5, np.nan, 2.1], [1.5, -4.2, np.nan], [0.0, 0.5, -1.0]],
        dtype=np.float64,
    )
    q = np.array([15.0, 85.0], dtype=np.float64)
    axis = (-2,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.array(
        [[1.0, 2.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float64
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": q,
                "axis": axis,
                "overwrite_input": overwrite_input,
                "method": method,
                "keepdims": keepdims,
                "weights": weights,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.nanpercentile_2"] = nanpercentile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanpercentile_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanpercentile_2'.")


check_valid('jax.numpy.nanpercentile', generated_inputs['jax.numpy.nanpercentile_2'], lib="jax", suffix=2)
