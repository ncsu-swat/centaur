
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanpercentile_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array, 3 percentiles
    a = np.array([1.0, np.nan, 3.0, 4.0, 5.0], dtype=np.float32)
    q = np.array([25.0, 50.0, 75.0], dtype=np.float32)
    axis = 0
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 1.0, 1.0, 2.0, 1.0], dtype=np.float32)
    
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

    # Input 2: 2D array, axis=1, keepdims=True
    a = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]], dtype=np.float32)
    q = np.array([50.0], dtype=np.float32)
    axis = 1
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float32)

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

    # Input 3: 2D array, axis=0, keepdims=False, multiple percentiles
    a = np.array([[np.nan, 2.0], [3.0, 4.0], [5.0, np.nan]], dtype=np.float32)
    q = np.array([10.0, 90.0], dtype=np.float32)
    axis = 0
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)

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

    # Input 4: 1D array with negative values and NaNs
    a = np.array([-10.0, -5.0, np.nan, 0.0, 5.0, 10.0], dtype=np.float32)
    q = np.array([50.0], dtype=np.float32)
    axis = 0
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

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

    # Input 5: 3D array of float64, axis=2, keepdims=True
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    a[0, 1, 2] = np.nan
    q = np.array([25.0, 75.0], dtype=np.float64)
    axis = 2
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([1.0, 1.0, 2.0, 1.0], dtype=np.float64)

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

    # Input 6: 4D array, axis=3
    a = np.ones((2, 2, 2, 2), dtype=np.float32) * np.nan
    a[0, 0, 0, 0] = 1.0
    a[0, 0, 0, 1] = 2.0
    q = np.array([50.0], dtype=np.float32)
    axis = 3
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 2.0], dtype=np.float32)

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

    # Input 7: 1D float64 with large values, q=0.0
    a = np.array([np.nan, 1e10, 2e10, 3e10], dtype=np.float64)
    q = np.array([0.0], dtype=np.float64)
    axis = 0
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

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

    # Input 8: 2D array, negative axis index (axis=-1)
    a = np.array([[1.5, np.nan, 2.5], [np.nan, 3.5, 4.5]], dtype=np.float32)
    q = np.array([100.0], dtype=np.float32)
    axis = -1
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)

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

    # Input 9: Array with multiple NaNs
    a = np.array([np.nan, np.nan, 5.0], dtype=np.float32)
    q = np.array([50.0], dtype=np.float32)
    axis = 0
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)

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

    # Input 10: 3D array, axis=1, multiple percentiles
    a = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    a[0, 0, 0] = np.nan
    q = np.array([30.0, 60.0], dtype=np.float32)
    axis = 1
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 1.0], dtype=np.float32)

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

generated_inputs["jax.numpy.nanpercentile_1"] = nanpercentile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanpercentile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanpercentile_1'.")


check_valid('jax.numpy.nanpercentile', generated_inputs['jax.numpy.nanpercentile_1'], lib="jax", suffix=1)
